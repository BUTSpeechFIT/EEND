# End-to-end Neural Diarization

PyTorch implementation for End-to-End Neural Diarization (EEND) based on Chainer implementation by [Hitachi](https://github.com/hitachi-speech/EEND). This code implements the encoder-decoder-based attractor with self-attention version of EEND.



## Usage
To run the training you can call:

    python eend/train.py -c examples/train.yaml
Note that in the example you need to define the train and validation data directories as well as the output directory. The rest of the parameters are standard ones, as used in our publication.
For fine-tuning, the process is similar:

    python eend/train.py -c examples/adapt.yaml
In that case, you will need to provide the path where to find the trained model(s) that you want to fine-tune.

To run the inference, you can call:

    python eend/infer.py -c examples/infer.yaml
Note that in the example you need to define the data, model and output directories.

## Citation
In case of using the software please cite:\
Federico Landini, Alicia Lozano-Diez, Mireia Diez, Lukáš Burget: [From Simulated Mixtures to Simulated Conversations as Training Data for End-to-End Neural Diarization](https://arxiv.org/abs/2204.00890)
```
@inproceedings{landini22_interspeech,
  author={Federico Landini and Alicia Lozano-Diez and Mireia Diez and Lukáš Burget},
  title={{From Simulated Mixtures to Simulated Conversations as Training Data for End-to-End Neural Diarization}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={5095--5099},
  doi={10.21437/Interspeech.2022-10451}
}
```

Federico Landini, Mireia Diez, Alicia Lozano-Diez, Lukáš Burget: [Multi-Speaker and Wide-Band Simulated Conversations as Training Data for End-to-End Neural Diarization](https://arxiv.org/abs/2211.06750)
```
@inproceedings{landini2023multi,
  title={Multi-Speaker and Wide-Band Simulated Conversations as Training Data for End-to-End Neural Diarization},
  author={Landini, Federico and Diez, Mireia and Lozano-Diez, Alicia and Burget, Luk{\'a}{\v{s}}},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```


## Contact
If you have any comment or question, please contact landini@fit.vutbr.cz
