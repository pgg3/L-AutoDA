import os
import argparse
import torch
from robustbench.utils import load_model

from core.data import get_data_by_id
from core.search_utils import AdversarialAttackTask

from evotoolkit import EoH
from evotoolkit.task.python_task import EoHPythonInterface
from evotoolkit.tools import HttpsApi

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ABS_PATH, "datasets")
MODEL_PATH = os.path.join(ABS_PATH, "models")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EoH to discover adversarial attack heuristics.")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "imagenet"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--atk_type", type=str, default="L2", choices=["L2", "Linf"])
    parser.add_argument("--model", type=str, default="Standard")
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--atk_step", type=int, default=300)
    parser.add_argument("--api_url", type=str, required=True, help="LLM API endpoint URL")
    parser.add_argument("--api_key", type=str, required=True, help="LLM API key")
    parser.add_argument("--llm_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--pop_size", type=int, default=5)
    parser.add_argument("--max_generations", type=int, default=10)
    parser.add_argument("--output_path", type=str, default="./results")

    args = parser.parse_args()
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
    interface = EoHPythonInterface(task)
    llm = HttpsApi(api_url=args.api_url, key=args.api_key, model=args.llm_model)

    eoh = EoH(
        interface=interface,
        running_llm=llm,
        output_path=args.output_path,
        max_generations=args.max_generations,
        pop_size=args.pop_size,
    )

    best = eoh.run()
    if best and best.evaluation_res:
        print(f"Best L2 distance: {-best.evaluation_res.score:.5f}")
        print(f"Best code:\n{best.sol_string}")
