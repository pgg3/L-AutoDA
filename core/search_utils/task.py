import sys
import types
import warnings

import numpy as np
import torch
import foolbox as fb

from evotoolkit.core import EvaluationResult, TaskSpec
from evotoolkit.task.python_task import PythonTask

from ..attacks import EvoAttack


class AdversarialAttackTask(PythonTask):
    """Evolves draw_proposals() functions for decision-based adversarial attacks.

    The function signature is:
        draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams) -> x_new

    Score = -L2_distance (evotoolkit maximizes; lower L2 = better attack).
    """

    def __init__(self, test_loader, model, atk_step: int, timeout_seconds: float = 300.0):
        self.test_loader = test_loader
        self.model = model
        self.atk_step = atk_step
        super().__init__(
            data={"test_loader": test_loader, "model": model, "atk_step": atk_step},
            timeout_seconds=timeout_seconds,
        )

    def build_python_spec(self, data) -> TaskSpec:
        prompt = (
            "Given an image 'org_img', its adversarial image 'best_adv_img', "
            "and a random normal noise 'std_normal_noise', "
            "design an algorithm to combine them to search for a new adversarial example 'x_new'. "
            "'hyperparams' ranges from 0.5 to 1.5. It gets larger when "
            "this algorithm outputs more adversarial examples, and vice versa. "
            "It can be used to control the step size of the search. "
            "Operations you may use include: adding, subtracting, multiplying, dividing, "
            "dot product, and l2 norm computation. Design a novel algorithm with various search techniques.\n\n"
            "Implement the following Python function:\n"
            "```python\n"
            "import numpy as np\n\n"
            "def draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams):\n"
            "    # org_img, best_adv_img, std_normal_noise, x_new: shape (3, img_height, img_width), bounds [0, 1]\n"
            "    # std_normal_noise: random normal noise\n"
            "    # hyperparams: numpy array of shape (1,)\n"
            "    # All inputs are numpy arrays. Only use numpy (imported as np).\n"
            "    ...\n"
            "    return x_new\n"
            "```"
        )
        return TaskSpec(
            name="adversarial_attack",
            prompt=prompt,
            modality="python",
        )

    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                heuristic_module = types.ModuleType("heuristic_module")
                exec(candidate_code, heuristic_module.__dict__)
                sys.modules[heuristic_module.__name__] = heuristic_module

                if not hasattr(heuristic_module, "draw_proposals"):
                    return EvaluationResult(
                        valid=False,
                        score=float("-inf"),
                        additional_info={"error": "Function 'draw_proposals' not defined"},
                    )

                l2_distance = self._run_attack(heuristic_module)
                return EvaluationResult(
                    valid=True,
                    score=-l2_distance,
                    additional_info={"l2_distance": l2_distance},
                )
        except Exception as e:
            return EvaluationResult(
                valid=False,
                score=float("-inf"),
                additional_info={"error": str(e)},
            )

    def _run_attack(self, heuristic_module) -> float:
        if torch.cuda.is_available():
            self.model.cuda()

        fmodel = fb.PyTorchModel(self.model, bounds=(0, 1))
        attack = EvoAttack(heuristic_module, steps=self.atk_step)

        for x, y in self.test_loader:
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            img_adv = attack.run(fmodel, x, y)
            distance = torch.linalg.norm((x - img_adv).flatten(start_dim=1), axis=1)
            return float(distance.mean().cpu().numpy())

        raise RuntimeError("test_loader is empty")
