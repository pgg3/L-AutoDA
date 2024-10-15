import numpy as np
import matplotlib.pyplot as plt
import importlib
import time
import pickle
import concurrent.futures
import copy
import torch
import foolbox as fb
import os
import sys
import types
import warnings

from .prompts import GetPrompts

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
ABS_PATH = os.path.dirname(os.path.abspath(__file__))

from ..attacks import EvoAttack


class Evaluation():
    def __init__(self, test_loader, model, atk_step) -> None:
        self.test_loader = test_loader
        self.model = model
        self.steps = atk_step
        self.running_time = 10
        self.prompts = GetPrompts()

    # @func_set_timeout(5)
    def greedy(self, eva):
        if torch.cuda.is_available():
            self.model.cuda()

        fmodel = fb.PyTorchModel(self.model, bounds=(0, 1))
        attack_use = EvoAttack(eva, steps=self.steps)
        for i, (x, y) in enumerate(self.test_loader):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            img_adv = attack_use.run(fmodel, x, y)
            # y_pred = self.model(x)
            # y_pred_adv = self.model(img_adv)
            distance = torch.linalg.norm((x - img_adv).flatten(start_dim=1), axis=1)
            distance = distance.mean()
            distance = float(distance.cpu().numpy())
            # print("average dis: ", distance)
            return distance

    def evaluate(self, code_string):
        try:
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Create a new module object
                heuristic_module = types.ModuleType("heuristic_module")

                # Execute the code string in the new module's namespace
                exec(code_string, heuristic_module.__dict__)

                # Add the module to sys.modules so it can be imported
                sys.modules[heuristic_module.__name__] = heuristic_module

                fitness = self.greedy(heuristic_module)

                return fitness
        except Exception as e:
            # print("Error:", str(e))
            return None
    # def evaluate(self, code_string):
    #     time.sleep(1)
    #     try:
    #         heuristic_module = importlib.import_module("ael_alg")
    #         eva = importlib.reload(heuristic_module)
    #         fitness = self.greedy(eva)
    #         return fitness
    #     except Exception as e:
    #         print("Error:", str(e))
    #         return None
