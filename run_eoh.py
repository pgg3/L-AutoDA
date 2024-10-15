import os
import torch
import argparse
from eoh import eoh
from eoh.utils.getParas import Paras
from core.search_utils.problem import Evaluation
from core.data import get_data_by_id
from robustbench.utils import load_model

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ABS_PATH, 'datasets')
MODEL_PATH = os.path.join(ABS_PATH, 'models')
DATA_CFG_PATH = os.path.join(ABS_PATH, 'configs', 'data_cfgs')

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run the EOH algorithm with specified parameters.")
    parser.add_argument('--dataset', type=str, default="cifar10", choices=["cifar10", "imagenet"])
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--atk_type', type=str, default="L2", choices=["L2", "Linf"])
    parser.add_argument('--model', type=str, default='Standard', help='Model architecture')
    parser.add_argument('--use_cuda', type=bool, default=True, help='Whether or not to use GPU')
    parser.add_argument('--atk_step', type=int, default=300, help='Number of attack steps')


    args = parser.parse_args()
    dataset = args.dataset
    batch_size = args.batch_size
    atk_type = args.atk_type
    atk_step = args.atk_step
    use_cuda = args.use_cuda
    use_cuda = use_cuda and torch.cuda.is_available()


    test_set = get_data_by_id(dataset, use_train_data=False, data_path=os.path.join(DATA_PATH, dataset))
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    use_model = load_model(
        model_name=args.model,
        model_dir=MODEL_PATH,
        dataset=dataset,
        threat_model=atk_type
    )
    if use_cuda:
        use_model.cuda()


    # Parameter initilization #
    paras = Paras()


    # Set your local problem
    problem_local = Evaluation(test_loader, use_model, atk_step)

    # Set parameters #
    paras.set_paras(
        method = "eoh",    # ['ael','eoh']
        problem = problem_local, # Set local problem, else use default problems
        llm_api_endpoint = "XXX", # set your LLM endpoint
        llm_api_key = "XXX",   # set your key
        llm_model = "gpt-3.5-turbo",
        ec_pop_size = 5, # number of samples in each population
        ec_n_pop = 10,  # number of populations
        exp_n_proc = 1,  # multi-core parallel
        exp_debug_mode = False,
        eva_numba_decorator = False,
        eva_timeout = 300
    )

    # initilization
    evolution = eoh.EVOL(paras)

    # run
    evolution.run()