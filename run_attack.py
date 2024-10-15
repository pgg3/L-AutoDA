import os
import torchvision
import timm
import torch
import json
import argparse
import torch.nn as nn
import importlib
import numpy as np
import foolbox as fb
from tqdm import tqdm
from torch.utils import data
from robustbench.utils import load_model

from core.data import get_data_by_id
from core.attacks import BoundaryAttack, HopSkipJumpAttack, AutoDAttack, EvoAttack


ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ABS_PATH, 'datasets')
MODEL_PATH = os.path.join(ABS_PATH, 'models')
DATA_CFG_PATH = os.path.join(ABS_PATH, 'configs', 'data_cfgs')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model training script")
    parser.add_argument('--dataset', type=str, default="cifar10", choices=["cifar10", "imagenet"])
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--model', type=str, default='Standard', help='Model architecture')
    parser.add_argument('--use_cuda', type=bool, default=True, help='Whether or not to use GPU')
    parser.add_argument('--atk_type', type=str, default="L2", choices=["L2", "Linf"])
    parser.add_argument('--atk_budget', type=int, default=1000, help='Attack budget')

    args = parser.parse_args()
    use_cuda = args.use_cuda
    use_cuda = use_cuda and torch.cuda.is_available()
    dataset = args.dataset
    batch_size = args.batch_size
    atk_type = args.atk_type
    atk_budget = args.atk_budget

    test_set = get_data_by_id(dataset, use_train_data=False, data_path=os.path.join(DATA_PATH, dataset))
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    data_cfg_file = os.path.join(DATA_CFG_PATH, dataset+".json")
    with open(data_cfg_file, 'r') as f:
        data_cfg = json.load(f)

    use_model = load_model(
        model_name=args.model,
        model_dir=MODEL_PATH,
        dataset=dataset,
        threat_model=atk_type
    )
    if use_cuda:
        use_model.cuda()

    fmodel = fb.PyTorchModel(use_model, bounds=data_cfg['bounds'])

    attack_use = BoundaryAttack(steps=atk_budget)
    # attack_use = HopSkipJumpAttack(max_queries=atk_budget)
    # attack_use = AutoDAttack(steps=atk_budget)

    # # put the searched function under ./heur/draw_new_np_no_mem.py
    # heuristic_module = importlib.import_module("heur.draw_new_np_no_mem")
    # eva = importlib.reload(heuristic_module)
    # attack_use = EvoAttack(eva, steps=atk_budget)

    attack_tqdm  = tqdm(test_loader)
    for (i, data) in enumerate(attack_tqdm):
        img, labels = data
        if use_cuda:
            img = img.cuda()
            labels = labels.cuda()
        img_adv = attack_use.run(fmodel, img, labels)
        # print(img_adv)
        with torch.no_grad():
            y_pred = use_model(img)
            y_pred_adv = use_model(img_adv)
            acc = torch.mean((torch.argmax(y_pred, dim=1) == labels).float())
            acc_adv = torch.mean((torch.argmax(y_pred_adv, dim=1) == labels).float())
            print("acc: {}, acc_adv: {}".format(acc, acc_adv))
            print(torch.linalg.norm((img - img_adv).flatten(start_dim=1), axis=1).mean())
        break
