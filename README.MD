# L-AutoDA

**L-AutoDA: Large Language Models for Automatically Evolving Decision-based Adversarial Attacks**

This is the official code repository for the GECCO '24 Companion paper. L-AutoDA uses [EvoToolkit](https://github.com/pgg3/evotoolkit) to automatically evolve decision-based adversarial attack algorithms via LLM-driven evolutionary search.

> The original standalone implementation is preserved on the [`v1`](../../tree/v1) branch.

## Installation

Requires Python >= 3.11. Install [uv](https://docs.astral.sh/uv/) if not already available:

```shell
pip install uv
```

Install all dependencies (PyTorch cu128 is resolved automatically):

```shell
uv sync
```

## Usage

### Run standalone attacks (no LLM)

```shell
uv run python run_attack.py --dataset cifar10 --model Standard --atk_type L2 --atk_budget 1000
```

### Run EvoEngineer to discover attack heuristics

```shell
uv run python run.py \
  --api_url https://api.openai.com/v1/chat/completions \
  --api_key YOUR_API_KEY \
  --llm_model gpt-3.5-turbo \
  --dataset cifar10 \
  --atk_step 300 \
  --pop_size 5 \
  --max_generations 10 \
  --output_path ./results
```

You can also configure the LLM endpoint via a `.env` file:

```
API_URL=https://api.openai.com/v1/chat/completions
API_KEY=your-api-key
MODEL=gpt-3.5-turbo
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{ping2024lautoda,
  author    = {Guo, Ping and Liu, Fei and Lin, Xi and Zhao, Qingchuan and Zhang, Qingfu},
  title     = {L-AutoDA: Large Language Models for Automatically Evolving Decision-based Adversarial Attacks},
  year      = {2024},
  isbn      = {9798400704956},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  url       = {https://doi.org/10.1145/3638530.3664121},
  doi       = {10.1145/3638530.3664121},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference Companion},
  pages     = {1846--1854},
  numpages  = {9},
  keywords  = {large language models, adversarial attacks, automated algorithm design, evolutionary algorithms},
  location  = {Melbourne, VIC, Australia},
  series    = {GECCO '24 Companion}
}
```
