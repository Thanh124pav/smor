"""OnlineReweighter — the bilevel reweighting loop (PLAN.md Stage B, §6, §13).

Given a :class:`WeightedLearner` and a fixed :class:`GroupAssignment`, alternate:

    train theta for R steps under current beta
      -> estimate the group hypergradients h_j^{(K)}
      -> update beta on the simplex
      -> continue

Logs a full time series of data-utility evidence and returns a :class:`ReweightingEvidence`
object that a future acquisition/scaling module can consume without knowing the learner.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from smor.evidence import ReweightingEvidence
from smor.learners.base import WeightedLearner
from smor.reweighting.beta import BetaDistribution
from smor.reweighting.config import OnlineReweighterConfig
from smor.reweighting.grouping import GroupAssignment
from smor.reweighting.hypergradient import group_hypergradient
from smor.reweighting.outer_objective import OuterObjective, ValidationLoss
from smor.reweighting.scheduler import ReweightScheduler
from smor.utils.logging import JsonlLogger, RunMetadata
from smor.utils.seeding import resolve_device


class OnlineReweighter:
    def __init__(self, config: OnlineReweighterConfig):
        self.config = config.validate()
        self.device = resolve_device(config.device)

    def fit(
        self,
        learner: WeightedLearner,
        group_assignment: GroupAssignment,
        outer_objective: Optional[OuterObjective] = None,
        logger: Optional[JsonlLogger] = None,
        eval_every: int = 5,
        eval_episodes: int = 64,
        checkpoint_path: Optional[str] = None,
        env_name: str = "point_mass",
    ) -> ReweightingEvidence:
        cfg = self.config
        outer_objective = outer_objective or ValidationLoss()
        M = group_assignment.num_groups
        gids = list(range(M))

        beta = BetaDistribution(
            num_groups=M,
            update=cfg.beta_update,
            lr=cfg.beta_lr,
            temperature=cfg.temperature,
            floor=cfg.beta_floor,
            entropy_reg=cfg.entropy_reg,
            score_clip=cfg.score_clip,
            standardize=cfg.beta_standardize,
            device=str(self.device),
        )
        scheduler = ReweightScheduler(cfg.reweight_interval, cfg.n_beta_updates, cfg.warmup_steps)

        # --- run metadata (§13) ---
        fid_counts = {int(f): int((group_assignment.group_fidelity == f).sum())
                      for f in np.unique(group_assignment.group_fidelity)}
        if logger is not None:
            meta = RunMetadata(
                seed=cfg.seed, config=cfg.to_dict(), env=env_name,
                n=cfg.n, K=cfg.K, damping=cfg.damping, neumann_lr=cfg.neumann_lr,
                beta_lr=cfg.beta_lr, reweight_interval=cfg.reweight_interval,
                n_demos=int(group_assignment.num_trajectories),
                fidelity_counts=fid_counts, device=str(self.device),
            )
            logger.log("run_metadata", **meta.to_dict())

        beta_history = [beta.as_numpy().copy()]
        hg_history: list[np.ndarray] = []
        eval_history: dict = defaultdict(list)
        compute_history: dict = defaultdict(list)
        policy_steps = 0

        # initial evaluation
        self._evaluate(learner, eval_episodes, eval_history, update=-1,
                       policy_steps=policy_steps, logger=logger)

        for kind, idx in scheduler:
            if kind == "train":
                weights = beta.weight_dict(gids)
                batches = learner.sample_batches(gids)
                metrics = learner.train_step(weights, batches)
                policy_steps += 1
                if logger is not None and policy_steps % max(1, cfg.reweight_interval) == 0:
                    logger.log("train_step", step=policy_steps, **metrics)
                continue

            # --- beta update (reweight event) ---
            t0 = time.perf_counter()
            batches = learner.sample_batches(gids)
            group_losses = learner.per_group_losses(batches)
            outer_loss = outer_objective.loss(learner)

            weights_f = beta.weight_dict(gids)
            inner_loss = None
            if cfg.K > 1:
                inner_loss = None
                for gid in gids:
                    term = weights_f[gid] * group_losses[gid]
                    inner_loss = term if inner_loss is None else inner_loss + term

            hg, hmeta = group_hypergradient(
                group_losses, outer_loss, learner.parameters_for_reweighting(),
                K=cfg.K, neumann_lr=cfg.neumann_lr, damping=cfg.damping,
                inner_loss=inner_loss, hvp_clip=cfg.hvp_clip,
                fallback_k1_on_invalid=cfg.fallback_k1_on_invalid, return_meta=True,
            )
            hvp_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            h_vec = np.array([hg[g] for g in gids], dtype=np.float64)
            info = beta.step(h_vec)
            beta_update_time = time.perf_counter() - t1

            beta_history.append(info["beta"].copy())
            hg_history.append(h_vec.copy())
            compute_history["hvp_time"].append(hvp_time)
            compute_history["beta_update_time"].append(beta_update_time)
            compute_history["hvp_count"].append(hmeta["hvp_count"])
            compute_history["n_fallbacks"].append(hmeta["n_fallbacks"])
            compute_history["policy_steps"].append(policy_steps)

            group_loss_vals = {g: float(group_losses[g].detach()) for g in gids}
            if logger is not None:
                logger.log(
                    "beta_update", update=idx, policy_steps=policy_steps,
                    beta=info["beta"], raw_hypergrad=h_vec, score=info["score"],
                    group_loss=[group_loss_vals[g] for g in gids],
                    outer_loss=float(outer_loss.detach()),
                    group_fidelity=group_assignment.group_fidelity.tolist(),
                    beta_entropy=info["entropy"], beta_l1_movement=info["beta_l1_movement"],
                    beta_l2_movement=info["beta_l2_movement"],
                    hvp_count=hmeta["hvp_count"], n_fallbacks=hmeta["n_fallbacks"],
                    g_out_norm=hmeta["g_out_norm"], hvp_time=hvp_time,
                    beta_update_time=beta_update_time,
                )

            if (idx + 1) % max(1, eval_every) == 0:
                self._evaluate(learner, eval_episodes, eval_history, update=idx,
                               policy_steps=policy_steps, logger=logger)

        # final evaluation
        self._evaluate(learner, eval_episodes, eval_history, update=cfg.n_beta_updates,
                       policy_steps=policy_steps, logger=logger)

        final_policy_path = None
        if checkpoint_path is not None:
            final_policy_path = str(Path(checkpoint_path))
            Path(final_policy_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(learner.state_dict(), final_policy_path)

        group_stats = self._group_stats(group_assignment, np.stack(beta_history),
                                        np.stack(hg_history) if hg_history else np.zeros((0, M)))

        return ReweightingEvidence(
            beta_history=np.stack(beta_history),
            hypergradient_history=np.stack(hg_history) if hg_history else np.zeros((0, M)),
            group_assignments=group_assignment.group_id.copy(),
            group_fidelity=group_assignment.group_fidelity.copy(),
            group_stats=group_stats,
            eval_history=dict(eval_history),
            compute_history=dict(compute_history),
            config=cfg.to_dict(),
            final_policy_path=final_policy_path,
        )

    # ---- helpers --------------------------------------------------------
    def _evaluate(self, learner, eval_episodes, eval_history, update, policy_steps, logger):
        metrics = learner.evaluate(n_episodes=eval_episodes)
        eval_history["update"].append(int(update))
        eval_history["policy_steps"].append(int(policy_steps))
        for k, v in metrics.items():
            eval_history[k].append(float(v))
        if logger is not None:
            logger.log("evaluation", update=update, policy_steps=policy_steps, **metrics)
        return metrics

    @staticmethod
    def _group_stats(ga: GroupAssignment, beta_hist: np.ndarray, hg_hist: np.ndarray) -> dict:
        M = ga.num_groups
        stats = {
            "group_sizes": ga.group_sizes.tolist(),
            "group_fidelity": ga.group_fidelity.tolist(),
            "final_beta": beta_hist[-1].tolist(),
            "mean_beta": beta_hist.mean(axis=0).tolist(),
        }
        if hg_hist.shape[0] > 0:
            stats["mean_hypergrad"] = hg_hist.mean(axis=0).tolist()
        return stats
