# L-AutoDA

L-AutoDA uses [evotoolkit](https://github.com/pgg3/evotoolkit) to automatically evolve decision-based adversarial attack algorithms via LLM-driven Evolution of Heuristics (EoH).

> The original standalone implementation is preserved on the [`v1`](../../tree/v1) branch.

## Installation

We suggest Python >= 3.11 to avoid compatibility issues with Robustbench.

```shell
conda create -n lautoda python=3.11
conda activate lautoda
```

Install PyTorch (adjust cuda version as needed):

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install remaining dependencies:

```shell
pip install evotoolkit
pip install git+https://github.com/RobustBench/robustbench.git
pip install foolbox eagerpy
```

## Usage

### Run standalone attacks (no LLM)

```shell
python run_attack.py --dataset cifar10 --model Standard --atk_type L2 --atk_budget 1000
```

### Run EoH to discover attack heuristics

```shell
python run_eoh.py \
  --api_url https://api.openai.com/v1/chat/completions \
  --api_key YOUR_API_KEY \
  --llm_model gpt-3.5-turbo \
  --dataset cifar10 \
  --atk_step 300 \
  --pop_size 5 \
  --max_generations 10 \
  --output_path ./results
```

Results are saved under `./results/` (evotoolkit format: `run_state.json`, `history/`, `summary/`).

## Reference

If you find this repo useful, please consider citing our paper:

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
