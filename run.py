import os
import argparse
import torch
from dotenv import load_dotenv
from robustbench.utils import load_model

from core.data import get_data_by_id
from core.search_utils import AdversarialAttackTask

from evotoolkit import EvoEngineer
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.tools import HttpsApi

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ABS_PATH, "datasets")
MODEL_PATH = os.path.join(ABS_PATH, "models")


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run EvoEngineer to discover adversarial attack heuristics.")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "imagenet"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--atk_type", type=str, default="L2", choices=["L2", "Linf"])
    parser.add_argument("--model", type=str, default="Standard")
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--atk_step", type=int, default=300)
    parser.add_argument("--api_url", type=str, default=None, help="LLM API endpoint URL (or set API_URL in .env)")
    parser.add_argument("--api_key", type=str, default=None, help="LLM API key (or set API_KEY in .env)")
    parser.add_argument("--llm_model", type=str, default=None, help="LLM model name (or set MODEL in .env, default: gpt-3.5-turbo)")
    parser.add_argument("--pop_size", type=int, default=5)
    parser.add_argument("--max_generations", type=int, default=10)
    parser.add_argument("--num_samplers", type=int, default=4)
    parser.add_argument("--num_evaluators", type=int, default=4)
    parser.add_argument("--output_path", type=str, default="./results")

    args = parser.parse_args()

    api_url = args.api_url or os.environ["API_URL"]
    api_key = args.api_key or os.environ["API_KEY"]
    llm_model = args.llm_model or os.environ.get("MODEL", "gpt-3.5-turbo")

    use_cuda = args.use_cuda and torch.cuda.is_available()

    # Load dataset and model
    test_set = get_data_by_id(
        args.dataset, use_train_data=False, data_path=os.path.join(DATA_PATH, args.dataset)
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True
    )
    use_model = load_model(
        model_name=args.model,
        model_dir=MODEL_PATH,
        dataset=args.dataset,
        threat_model=args.atk_type,
    )
    if use_cuda:
        use_model.cuda()

    # Build evotoolkit components
    task = AdversarialAttackTask(test_loader, use_model, args.atk_step)
    interface = EvoEngineerPythonInterface(task)
    llm = HttpsApi(api_url=api_url, key=api_key, model=llm_model)

    algo = EvoEngineer(
        interface=interface,
        running_llm=llm,
        output_path=args.output_path,
        max_generations=args.max_generations,
        pop_size=args.pop_size,
        num_samplers=args.num_samplers,
        num_evaluators=args.num_evaluators,
    )

    best = algo.run()
    if best and best.evaluation_res:
        print(f"Best L2 distance: {-best.evaluation_res.score:.5f}")
        print(f"Best code:\n{best.sol_string}")
