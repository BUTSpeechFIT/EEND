#!/usr/bin/env python3

# Copyright 2022 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.

import torch.optim as optim
from torch.nn import Module
from types import SimpleNamespace
from typing import Any, Dict


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size: int, warmup: int, optimizer: optim) -> None:
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value
            for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state.
            Should be an object returned from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self) -> None:
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step: int = None) -> float:
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return (
            self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def get_rate(self) -> float:
        return self._rate

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()


def setup_optimizer(args: SimpleNamespace, model: Module) -> optim:
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'noam':
        optimizer = NoamOpt(
            args.hidden_size,
            args.noam_warmup_steps,
            optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    else:
        raise ValueError(args.optimizer)
    return optimizer


def get_rate(optimizer: optim) -> float:
    if isinstance(optimizer, NoamOpt):
        return optimizer.get_rate()
    else:
        for param_group in optimizer.param_groups:
            return param_group['lr']
