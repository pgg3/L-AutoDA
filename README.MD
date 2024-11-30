# L-AutoDA

# Installation and Setup
We suggest to use python version >= 3.11 to avoid bugs (`Robustbench` use python dependency>=3.10). Use the following command to install the required packages:

```shell
conda create -n LDAEX python=3.11
```

## Install torch
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Install Robustbench
```shell
pip install git+https://github.com/RobustBench/robustbench.git
```

## Install Other Dependencies
```shell
pip install foolbox matplotlib joblib
```


## Reference
If you find this repo useful, please consider to cite our paper:

```
@inproceedings{guo:2024:lautoda,
  author       = {Ping Guo and
                  Fei Liu and
                  Xi Lin and
                  Qingchuan Zhao and
                  Qingfu Zhang},
  title        = {L-AutoDA: Large Language Models for Automatically Evolving Decision-based
                  Adversarial Attacks},
  booktitle    = {Proceedings of the Genetic and Evolutionary Computation Conference
                  Companion, {GECCO}},
  pages        = {1846--1854},
  publisher    = {{ACM}},
  year         = {2024}
}
```

