from typing import Union, Tuple, Optional, Any
from typing_extensions import Literal
import numpy as np
import eagerpy as ep
import logging

from foolbox.devutils import flatten
from foolbox.devutils import atleast_kd

from foolbox.types import Bounds

from foolbox.models import Model

from foolbox.criteria import Criterion

from foolbox.distances import l2

from foolbox.tensorboard import TensorBoard

from foolbox.attacks.blended_noise import LinearSearchBlendedUniformNoiseAttack

from foolbox.attacks.base import MinimizationAttack
from foolbox.attacks.base import T
from foolbox.attacks.base import get_criterion
from foolbox.attacks.base import get_is_adversarial
from foolbox.attacks.base import raise_if_kwargs
from foolbox.attacks.base import verify_input_bounds


class BoundaryAttack(MinimizationAttack):
    distance = l2
    # can only perform l2 attacks
    def __init__(
        self,
        init_attack: Optional[MinimizationAttack] = None,
        steps: int = 25000,
        spherical_step: float = 1e-2,
        source_step: float = 1e-2,
        source_step_convergance: float = 1e-7,
        step_adaptation: float = 1.5,
        tensorboard: Union[Literal[False], None, str] = False,
        update_stats_every_k: int = 10,
    ):
        if init_attack is not None and not isinstance(init_attack, MinimizationAttack):
            raise NotImplementedError
        self.init_attack = init_attack
        self.steps = steps
        self.spherical_step = spherical_step
        self.source_step = source_step
        self.source_step_convergance = source_step_convergance
        self.step_adaptation = step_adaptation
        self.tensorboard = tensorboard
        self.update_stats_every_k = update_stats_every_k

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        *,
        early_stop: Optional[float] = None,
        starting_points: Optional[T] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        originals, restore_type = ep.astensor_(inputs)
        del inputs, kwargs

        verify_input_bounds(originals, model)

        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)

        if starting_points is None:
            init_attack: MinimizationAttack
            if self.init_attack is None:
                init_attack = LinearSearchBlendedUniformNoiseAttack(steps=50)
                logging.info(
                    f"Neither starting_points nor init_attack given. Falling"
                    f" back to {init_attack!r} for initialization."
                )
            else:
                init_attack = self.init_attack
            # TODO: use call and support all types of attacks (once early_stop is
            # possible in __call__)
            best_advs = init_attack.run(
                model, originals, criterion, early_stop=early_stop
            )
        else:
            best_advs = ep.astensor(starting_points)

        is_adv = is_adversarial(best_advs)
        if not is_adv.all():
            failed = is_adv.logical_not().float32().sum()
            if starting_points is None:
                raise ValueError(
                    f"init_attack failed for {failed} of {len(is_adv)} inputs"
                )
            else:
                raise ValueError(
                    f"{failed} of {len(is_adv)} starting_points are not adversarial"
                )
        del starting_points

        N = len(originals)
        ndim = originals.ndim
        spherical_steps = ep.ones(originals, N) * self.spherical_step
        source_steps = ep.ones(originals, N) * self.source_step

        # create two queues for each sample to track success rates
        # (used to update the hyper parameters)
        stats_spherical_adversarial = ArrayQueue(maxlen=100, N=N)
        stats_step_adversarial = ArrayQueue(maxlen=30, N=N)

        bounds = model.bounds

        for step in range(1, self.steps + 1):
            converged = source_steps < self.source_step_convergance
            if converged.all():
                break  # pragma: no cover
            converged = atleast_kd(converged, ndim)

            # only check spherical candidates every k steps
            check_spherical_and_update_stats = step % self.update_stats_every_k == 0

            candidates, spherical_candidates = draw_proposals(
                bounds,
                originals,
                best_advs,
                spherical_steps,
                source_steps,
            )

            is_adv = is_adversarial(candidates)

            spherical_is_adv: Optional[ep.Tensor]
            if check_spherical_and_update_stats:
                spherical_is_adv = is_adversarial(spherical_candidates)
                stats_spherical_adversarial.append(spherical_is_adv)
                stats_step_adversarial.append(is_adv)
            else:
                spherical_is_adv = None

            # in theory, we are closer per construction
            # but limited numerical precision might break this
            distances = ep.norms.l2(flatten(originals - candidates), axis=-1)
            unnormalized_source_directions = originals - best_advs
            source_norms = ep.norms.l2(flatten(unnormalized_source_directions), axis=-1)
            closer = distances < source_norms
            is_best_adv = ep.logical_and(is_adv, closer)
            is_best_adv = atleast_kd(is_best_adv, ndim)

            cond = converged.logical_not().logical_and(is_best_adv)
            best_advs = ep.where(cond, candidates, best_advs)

            if check_spherical_and_update_stats:
                full = stats_spherical_adversarial.isfull()
                if full.any():
                    probs = stats_spherical_adversarial.mean()
                    cond1 = ep.logical_and(probs > 0.5, full)
                    spherical_steps = ep.where(
                        cond1, spherical_steps * self.step_adaptation, spherical_steps
                    )
                    source_steps = ep.where(
                        cond1, source_steps * self.step_adaptation, source_steps
                    )
                    cond2 = ep.logical_and(probs < 0.2, full)
                    spherical_steps = ep.where(
                        cond2, spherical_steps / self.step_adaptation, spherical_steps
                    )
                    source_steps = ep.where(
                        cond2, source_steps / self.step_adaptation, source_steps
                    )
                    stats_spherical_adversarial.clear(ep.logical_or(cond1, cond2))

                full = stats_step_adversarial.isfull()
                if full.any():
                    probs = stats_step_adversarial.mean()
                    # TODO: algorithm: changed the two values because we are currently tracking p(source_step_sucess)
                    # instead of p(source_step_success | spherical_step_sucess) that was tracked before
                    cond1 = ep.logical_and(probs > 0.25, full)
                    source_steps = ep.where(
                        cond1, source_steps * self.step_adaptation, source_steps
                    )
                    cond2 = ep.logical_and(probs < 0.1, full)
                    source_steps = ep.where(
                        cond2, source_steps / self.step_adaptation, source_steps
                    )
                    stats_step_adversarial.clear(ep.logical_or(cond1, cond2))
        return restore_type(best_advs)


class ArrayQueue:
    def __init__(self, maxlen: int, N: int):
        # we use NaN as an indicator for missing data
        self.data = np.full((maxlen, N), np.nan)
        self.next = 0
        # used to infer the correct framework because this class uses NumPy
        self.tensor: Optional[ep.Tensor] = None

    @property
    def maxlen(self) -> int:
        return int(self.data.shape[0])

    @property
    def N(self) -> int:
        return int(self.data.shape[1])

    def append(self, x: ep.Tensor) -> None:
        if self.tensor is None:
            self.tensor = x
        x = x.numpy()
        assert x.shape == (self.N,)
        self.data[self.next] = x
        self.next = (self.next + 1) % self.maxlen

    def clear(self, dims: ep.Tensor) -> None:
        if self.tensor is None:
            self.tensor = dims  # pragma: no cover
        dims = dims.numpy()
        assert dims.shape == (self.N,)
        assert dims.dtype == bool
        self.data[:, dims] = np.nan

    def mean(self) -> ep.Tensor:
        assert self.tensor is not None
        result = np.nanmean(self.data, axis=0)
        return ep.from_numpy(self.tensor, result)

    def isfull(self) -> ep.Tensor:
        assert self.tensor is not None
        result = ~np.isnan(self.data).any(axis=0)
        return ep.from_numpy(self.tensor, result)


def draw_proposals(
    bounds: Bounds,
    originals: ep.Tensor,
    perturbed: ep.Tensor,
    spherical_steps: ep.Tensor,
    source_steps: ep.Tensor,
) -> Tuple[ep.Tensor, ep.Tensor]:

    ndim = originals.ndim
    unnormalized_source_directions = originals - perturbed
    source_norms = ep.norms.l2(flatten(unnormalized_source_directions), axis=-1)
    source_directions = unnormalized_source_directions / atleast_kd(
        source_norms, ndim
    )

    # remember the actual shape
    shape = originals.shape
    assert perturbed.shape == shape
    assert unnormalized_source_directions.shape == shape
    assert source_directions.shape == shape

    # flatten everything to (batch, size)
    originals = flatten(originals)
    perturbed = flatten(perturbed)
    unnormalized_source_directions = flatten(unnormalized_source_directions)
    source_directions = flatten(source_directions)
    N, D = originals.shape

    assert source_norms.shape == (N,)
    assert spherical_steps.shape == (N,)
    assert source_steps.shape == (N,)

    # draw from an iid Gaussian (we can share this across the whole batch)
    eta = ep.normal(perturbed, (D, 1))

    # make orthogonal (source_directions are normalized)
    eta = eta.T - ep.matmul(source_directions, eta) * source_directions
    assert eta.shape == (N, D)

    # rescale
    norms = ep.norms.l2(eta, axis=-1)
    assert norms.shape == (N,)
    eta = eta * atleast_kd(spherical_steps * source_norms / norms, eta.ndim)

    # project on the sphere using Pythagoras
    distances = atleast_kd((spherical_steps.square() + 1).sqrt(), eta.ndim)
    directions = eta - unnormalized_source_directions
    spherical_candidates = originals + directions / distances

    # clip
    min_, max_ = bounds
    spherical_candidates = spherical_candidates.clip(min_, max_)

    # step towards the original inputs
    new_source_directions = originals - spherical_candidates
    assert new_source_directions.ndim == 2
    new_source_directions_norms = ep.norms.l2(flatten(new_source_directions), axis=-1)

    # length if spherical_candidates would be exactly on the sphere
    lengths = source_steps * source_norms

    # length including correction for numerical deviation from sphere
    lengths = lengths + new_source_directions_norms - source_norms

    # make sure the step size is positive
    lengths = ep.maximum(lengths, 0)

    # normalize the length
    lengths = lengths / new_source_directions_norms
    lengths = atleast_kd(lengths, new_source_directions.ndim)

    candidates = spherical_candidates + lengths * new_source_directions

    # clip
    candidates = candidates.clip(min_, max_)

    # restore shape
    candidates = candidates.reshape(shape)
    spherical_candidates = spherical_candidates.reshape(shape)
    return candidates, spherical_candidates