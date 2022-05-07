#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.


from backend.models import (
    average_checkpoints,
    get_model,
    load_checkpoint,
    save_checkpoint,
)
from backend.updater import setup_optimizer, get_rate
from common_utils.diarization_dataset import KaldiDiarizationDataset
from common_utils.gpu_utils import use_single_gpu
from common_utils.metrics import (
    calculate_metrics,
    new_metrics,
    reset_metrics,
    update_metrics,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
import numpy as np
import os
import random
import torch
import logging
import yamlargparse


def _init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def pad_sequence(
    features: List[torch.Tensor],
    labels: List[torch.Tensor],
    seq_len: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    features_padded = []
    labels_padded = []
    assert len(features) == len(labels), (
        f"Features and labels in batch were expected to match but got "
        "{len(features)} features and {len(labels)} labels.")
    for i, _ in enumerate(features):
        assert features[i].shape[0] == labels[i].shape[0], (
            f"Length of features and labels were expected to match but got "
            "{features[i].shape[0]} and {labels[i].shape[0]}")
        length = features[i].shape[0]
        if length < seq_len:
            extend = seq_len - length
            features_padded.append(torch.cat((features[i], -torch.ones((
                extend, features[i].shape[1]))), dim=0))
            labels_padded.append(torch.cat((labels[i], -torch.ones((
                extend, labels[i].shape[1]))), dim=0))
        elif length > seq_len:
            raise (f"Sequence of length {length} was received but only "
                   "{seq_len} was expected.")
        else:
            features_padded.append(features[i])
            labels_padded.append(labels[i])
    return features_padded, labels_padded


def _convert(
    batch: List[Tuple[torch.Tensor, torch.Tensor, str]]
) -> Dict[str, Any]:
    return {'xs': [x for x, _, _ in batch],
            'ts': [t for _, t, _ in batch],
            'names': [r for _, _, r in batch]}


def compute_loss_and_metrics(
    model: torch.nn.Module,
    labels: torch.Tensor,
    input: torch.Tensor,
    acum_metrics: Dict[str, float]
) -> Tuple[torch.Tensor, Dict[str, float]]:
    n_speakers = np.asarray([t.shape[1] for t in labels])
    y_pred, attractor_loss = model(input, labels, args)
    loss, standard_loss = model.get_loss(
        y_pred, labels, n_speakers, attractor_loss)
    metrics = calculate_metrics(
        labels.detach(), y_pred.detach(), threshold=0.5)
    acum_metrics = update_metrics(acum_metrics, metrics)
    acum_metrics['loss'] += loss.detach()
    acum_metrics['loss_standard'] += standard_loss.detach()
    acum_metrics['loss_attractor'] += attractor_loss.detach()
    return loss, acum_metrics


def get_training_dataloaders(
    args: SimpleNamespace
) -> Tuple[DataLoader, DataLoader]:
    train_set = KaldiDiarizationDataset(
        args.train_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        feature_dim=args.feature_dim,
        frame_shift=args.frame_shift,
        frame_size=args.frame_size,
        input_transform=args.input_transform,
        n_speakers=args.num_speakers,
        sampling_rate=args.sampling_rate,
        shuffle=args.time_shuffle,
        subsampling=args.subsampling,
        use_last_samples=args.use_last_samples,
        min_length=args.min_length,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.train_batchsize,
        collate_fn=_convert,
        num_workers=args.num_workers,
        shuffle=True,
        worker_init_fn=_init_fn,
    )

    dev_set = KaldiDiarizationDataset(
        args.valid_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        feature_dim=args.feature_dim,
        frame_shift=args.frame_shift,
        frame_size=args.frame_size,
        input_transform=args.input_transform,
        n_speakers=args.num_speakers,
        sampling_rate=args.sampling_rate,
        shuffle=args.time_shuffle,
        subsampling=args.subsampling,
        use_last_samples=args.use_last_samples,
        min_length=args.min_length,
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.dev_batchsize,
        collate_fn=_convert,
        num_workers=1,
        shuffle=False,
        worker_init_fn=_init_fn,
    )

    Y_train, _, _ = train_set.__getitem__(0)
    Y_dev, _, _ = dev_set.__getitem__(0)
    assert Y_train.shape[1] == Y_dev.shape[1], \
        f"Train features dimensionality ({Y_train.shape[1]}) and \
        dev features dimensionality ({Y_dev.shape[1]}) differ."
    assert Y_train.shape[1] == (
        args.feature_dim * (1 + 2 * args.context_size)), \
        f"Expected feature dimensionality of {args.feature_dim} \
        but {Y_train.shape[1]} found."

    return train_loader, dev_loader


def parse_arguments() -> SimpleNamespace:
    parser = yamlargparse.ArgumentParser(description='EEND training')
    parser.add_argument('-c', '--config', help='config file path',
                        action=yamlargparse.ActionConfigFile)
    parser.add_argument('--context-size', default=0, type=int)
    parser.add_argument('--dev-batchsize', default=1, type=int,
                        help='number of utterances in one development batch')
    parser.add_argument('--encoder-units', type=int,
                        help='number of units in the encoder')
    parser.add_argument('--feature-dim', type=int)
    parser.add_argument('--frame-shift', type=int)
    parser.add_argument('--frame-size', type=int)
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', default=-1, type=int,
                        help='gradient clipping. if < 0, no clipping')
    parser.add_argument('--hidden-size', type=int,
                        help='number of units in SA blocks')
    parser.add_argument('--init-epochs', type=str, default='',
                        help='Initialize model with average of epochs \
                        separated by commas or - for intervals.')
    parser.add_argument('--init-model-path', type=str, default='',
                        help='Initialize the model from the given directory')
    parser.add_argument('--input-transform', default='',
                        choices=['logmel', 'logmel_meannorm',
                                 'logmel_meanvarnorm'],
                        help='input normalization transform')
    parser.add_argument('--log-report-batches-num', default=1, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--max-epochs', type=int,
                        help='Max. number of epochs to train')
    parser.add_argument('--min-length', default=0, type=int,
                        help='Minimum number of frames for the sequences'
                             ' after downsampling.')
    parser.add_argument('--model-type', default='TransformerEDA',
                        help='Type of model (for now only TransformerEDA)')
    parser.add_argument('--noam-warmup-steps', default=100000, type=float)
    parser.add_argument('--num-frames', default=500, type=int,
                        help='number of frames in one utterance')
    parser.add_argument('--num-speakers', type=int,
                        help='maximum number of speakers allowed')
    parser.add_argument('--num-workers', default=1, type=int,
                        help='number of workers in train DataLoader')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--sampling-rate', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--subsampling', default=10, type=int)
    parser.add_argument('--train-batchsize', default=1, type=int,
                        help='number of utterances in one train batch')
    parser.add_argument('--train-data-dir',
                        help='kaldi-style data dir used for training.')
    parser.add_argument('--transformer-encoder-dropout', type=float)
    parser.add_argument('--transformer-encoder-n-heads', type=int)
    parser.add_argument('--transformer-encoder-n-layers', type=int)
    parser.add_argument('--use-last-samples', default=True, type=bool)
    parser.add_argument('--valid-data-dir',
                        help='kaldi-style data dir used for validation.')

    attractor_args = parser.add_argument_group('attractor')
    attractor_args.add_argument(
        '--time-shuffle', action='store_true',
        help='Shuffle time-axis order before input to the network')
    attractor_args.add_argument(
        '--attractor-loss-ratio', default=1.0, type=float,
        help='weighting parameter')
    attractor_args.add_argument(
        '--attractor-encoder-dropout', type=float)
    attractor_args.add_argument(
        '--attractor-decoder-dropout', type=float)
    attractor_args.add_argument(
        '--detach-attractor-loss', type=bool,
        help='If True, avoid backpropagation on attractor loss')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()

    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    logging.info(args)

    writer = SummaryWriter(f"{args.output_path}/tensorboard")

    train_loader, dev_loader = get_training_dataloaders(args)

    if args.gpu >= 1:
        gpuid = use_single_gpu(args.gpu)
        logging.info('GPU device {} is used'.format(gpuid))
        args.device = torch.device("cuda")
    else:
        gpuid = -1
        args.device = torch.device("cpu")

    if args.init_model_path == '':
        model = get_model(args)
        optimizer = setup_optimizer(args, model)
    else:
        model = get_model(args)
        model = average_checkpoints(
            args.device, model, args.init_model_path, args.init_epochs)
        optimizer = setup_optimizer(args, model)

    train_batches_qty = len(train_loader)
    dev_batches_qty = len(dev_loader)
    logging.info(f"#batches quantity for train: {train_batches_qty}")
    logging.info(f"#batches quantity for dev: {dev_batches_qty}")

    acum_train_metrics = new_metrics()
    acum_dev_metrics = new_metrics()

    if os.path.isfile(os.path.join(
            args.output_path, 'models', 'checkpoint_0.tar')):
        # Load latest model and continue from there
        directory = os.path.join(args.output_path, 'models')
        checkpoints = os.listdir(directory)
        paths = [os.path.join(directory, basename) for basename in checkpoints]
        latest = max(paths, key=os.path.getctime)
        epoch, model, optimizer, _ = load_checkpoint(args, latest)
        init_epoch = epoch + 1
    else:
        init_epoch = 0
        # Save initial model
        save_checkpoint(args, init_epoch, model, optimizer, 0)

    for epoch in range(init_epoch, args.max_epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            features = batch['xs']
            labels = batch['ts']
            features, labels = pad_sequence(features, labels, args.num_frames)
            features = torch.stack(features).to(args.device)
            labels = torch.stack(labels).to(args.device)
            loss, acum_train_metrics = compute_loss_and_metrics(
                model, labels, features, acum_train_metrics)
            if i % args.log_report_batches_num == \
                    (args.log_report_batches_num-1):
                for k in acum_train_metrics.keys():
                    writer.add_scalar(
                        f"train_{k}",
                        acum_train_metrics[k] / args.log_report_batches_num,
                        epoch * train_batches_qty + i)
                writer.add_scalar(
                    "lrate",
                    get_rate(optimizer),
                    epoch * train_batches_qty + i)
                acum_train_metrics = reset_metrics(acum_train_metrics)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradclip)
            optimizer.step()

        save_checkpoint(args, epoch+1, model, optimizer, loss)

        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(dev_loader):
                features = batch['xs']
                labels = batch['ts']
                features, labels = pad_sequence(
                    features, labels, args.num_frames)
                features = torch.stack(features).to(args.device)
                labels = torch.stack(labels).to(args.device)
                _, acum_dev_metrics = compute_loss_and_metrics(
                    model, labels, features, acum_dev_metrics)
        for k in acum_dev_metrics.keys():
            writer.add_scalar(
                f"dev_{k}", acum_dev_metrics[k] / dev_batches_qty,
                epoch * dev_batches_qty + i)
        acum_dev_metrics = reset_metrics(acum_dev_metrics)
