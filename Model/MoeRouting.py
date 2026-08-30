"""Deterministic token-to-expert routing for MoE workloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class RoutingResult:
    """Per-rank token counts after routing one microbatch."""

    matrix: np.ndarray
    received_tokens: np.ndarray
    dropped_tokens: int
    expert_loads: np.ndarray

    def sends_from(self, ep_local: int) -> np.ndarray:
        """Tokens this rank sends to each destination."""
        return self.matrix[ep_local]

    def receives_by(self, ep_local: int) -> np.ndarray:
        """Tokens this rank receives from each source."""
        return self.matrix[:, ep_local]


class MoeRouter:
    """Seeded token-to-expert router."""

    def __init__(
        self,
        num_experts: int,
        ep_size: int,
        top_k: int = 1,
        capacity_factor: float = 1.25,
        distribution: dict | None = None,
        placement: dict | None = None,
        seed: int = 0,
        resample: str = "per_microbatch",
        drop_tokens: bool = True,
    ):
        if num_experts < 1:
            raise ValueError(f"moe.num_experts must be >= 1 (got {num_experts})")
        if ep_size < 1:
            raise ValueError(f"moe.ep_size must be >= 1 (got {ep_size})")
        if top_k < 1:
            raise ValueError(f"moe.top_k must be >= 1 (got {top_k})")

        self.num_experts = int(num_experts)
        self.ep_size = int(ep_size)
        self.top_k = int(top_k)
        self.capacity_factor = float(capacity_factor)
        self.seed = int(seed)
        self.resample = str(resample)
        self.drop_tokens = bool(drop_tokens)

        if self.resample not in ("per_microbatch", "per_layer", "fixed"):
            raise ValueError(
                "moe.resample must be one of 'per_microbatch', 'per_layer', 'fixed' "
                f"(got {self.resample})"
            )

        self.distribution = dict(distribution or {"type": "uniform"})
        self.dist_type = str(self.distribution.get("type", "uniform"))
        if self.dist_type not in ("uniform", "dirichlet", "explicit", "lognormal"):
            raise ValueError(
                "moe.distribution.type must be one of 'uniform', 'dirichlet', "
                f"'explicit', 'lognormal' (got {self.dist_type})"
            )

        self.placement_cfg = dict(placement or {"strategy": "contiguous"})
        self.expert_to_rank = self._build_placement(self.placement_cfg)

    def _build_placement(self, cfg: dict) -> np.ndarray:
        """Map expert id to local EP rank."""
        strategy = str(cfg.get("strategy", "contiguous"))
        E, P = self.num_experts, self.ep_size

        if strategy == "custom":
            mapping = cfg.get("expert_to_rank")
            if mapping is None:
                raise ValueError(
                    "moe.placement.strategy='custom' requires 'expert_to_rank'"
                )
            if len(mapping) != E:
                raise ValueError(
                    f"moe.placement.expert_to_rank must have length num_experts={E} "
                    f"(got {len(mapping)})"
                )
            arr = np.asarray(mapping, dtype=np.int64)
            if arr.min() < 0 or arr.max() >= P:
                raise ValueError(
                    f"moe.placement.expert_to_rank entries must be in [0, {P - 1}]"
                )
            return arr

        if strategy == "round_robin":
            return np.array([e % P for e in range(E)], dtype=np.int64)

        if strategy == "contiguous":
            if E % P != 0:
                raise ValueError(
                    f"contiguous placement requires num_experts ({E}) divisible by "
                    f"ep_size ({P}); use strategy='custom' otherwise"
                )
            experts_per_rank = E // P
            return np.array([e // experts_per_rank for e in range(E)], dtype=np.int64)

        raise ValueError(
            "moe.placement.strategy must be 'contiguous', 'round_robin' or 'custom' "
            f"(got {strategy})"
        )

    def group_key(self, layer: int, microbatch: int, ep_block: int) -> Tuple[int, ...]:
        """Routing key at the configured resample granularity."""
        if self.resample == "fixed":
            return (int(ep_block),)
        if self.resample == "per_layer":
            return (int(layer), int(ep_block))
        return (int(layer), int(microbatch), int(ep_block))

    def _rng(self, key: Sequence[int], source_rank: int) -> np.random.Generator:
        entropy = [self.seed, *[int(k) for k in key], int(source_rank)]
        return np.random.default_rng(np.random.SeedSequence(entropy))

    def _probs(self, rng: np.random.Generator) -> np.ndarray:
        E = self.num_experts
        if self.dist_type == "uniform":
            return np.full(E, 1.0 / E)
        if self.dist_type == "dirichlet":
            alpha = float(self.distribution.get("alpha", 1.0))
            return rng.dirichlet(np.full(E, alpha))
        if self.dist_type == "lognormal":
            sigma = float(self.distribution.get("sigma", 1.0))
            w = rng.lognormal(mean=0.0, sigma=sigma, size=E)
            return w / w.sum()
        # explicit
        probs = np.asarray(self.distribution.get("probs", []), dtype=np.float64)
        if probs.size != E:
            raise ValueError(
                f"moe.distribution.probs must have length num_experts={E} "
                f"(got {probs.size})"
            )
        if probs.min() < 0:
            raise ValueError("moe.distribution.probs must be non-negative")
        total = probs.sum()
        if total <= 0:
            raise ValueError("moe.distribution.probs must sum to > 0")
        return probs / total

    def route(self, key: Sequence[int], tokens_per_rank: int) -> RoutingResult:
        """Route one microbatch for one EP group."""
        E, P = self.num_experts, self.ep_size
        tokens_per_rank = max(0, int(tokens_per_rank))

        # counts[s][e]: tokens from rank s to expert e, before capacity.
        counts = np.zeros((P, E), dtype=np.int64)
        slots = tokens_per_rank * self.top_k
        for s in range(P):
            if slots <= 0:
                break
            rng = self._rng(key, s)
            p = self._probs(rng)
            counts[s] = rng.multinomial(slots, p)

        dropped = 0
        if self.drop_tokens and P > 0:
            total_group_tokens = tokens_per_rank * P
            capacity = self.capacity_factor * total_group_tokens * self.top_k / E
            capacity = int(np.floor(capacity))
            agg = counts.sum(axis=0)
            for e in range(E):
                if agg[e] > capacity and agg[e] > 0:
                    keep_ratio = capacity / agg[e]
                    kept_col = np.floor(counts[:, e] * keep_ratio).astype(np.int64)
                    dropped += int(counts[:, e].sum() - kept_col.sum())
                    counts[:, e] = kept_col

        expert_loads = counts.sum(axis=0)

        # fold experts onto host ranks
        matrix = np.zeros((P, P), dtype=np.int64)
        for e in range(E):
            dest = int(self.expert_to_rank[e])
            matrix[:, dest] += counts[:, e]

        received = matrix.sum(axis=0)
        return RoutingResult(
            matrix=matrix,
            received_tokens=received,
            dropped_tokens=int(dropped),
            expert_loads=expert_loads,
        )
