from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List, Optional, Protocol, Sequence

import numpy as np


# -----------------------------
# Adapter Interfaces
# -----------------------------


class BasePersonalizedAdapter(Protocol):
    """Protocol for integrating existing plain-Python implementation.

    The Flower client calls into this adapter to perform all domain-specific
    computations. Replace DummyAdapter with a concrete implementation that wraps
    your code with minimal changes.
    """

    # ------ States and parameterization ------
    def init_global_state(self, *, seed: int) -> Any:
        ...

    def init_personalized_state(self, *, seed: int) -> Any:
        ...

    def global_state_to_ndarrays(self, state: Any) -> List[np.ndarray]:
        ...

    def ndarrays_to_global_state(self, arrays: Sequence[np.ndarray]) -> Any:
        ...

    def sync_personalized_with_global(self, *, personal_state: Any, global_state: Any) -> Any:
        """Optionally re-condition personalized posterior on the current global posterior."""
        ...

    # ------ Data utilities ------
    def get_default_train_data(self) -> Any:
        ...

    def get_default_test_data(self) -> Any:
        ...

    def num_examples(self, dataset: Any) -> int:
        ...

    def minibatch_iterator(self, dataset: Any, *, batch_size: int, rng: np.random.Generator) -> Iterator[Any]:
        ...

    # ------ Learning steps and evaluation ------
    @dataclass
    class StepOutput:
        updated_state: Any
        metrics: Optional[dict[str, float]] = None

    def step_personalized(
        self,
        *,
        personal_state: Any,
        global_state: Any,
        batch: Any,
        num_mc_samples: int,
        kl_weight: float,
        dataset_size: int,
        rng: np.random.Generator,
    ) -> StepOutput:
        ...

    def step_localized_global(
        self,
        *,
        personal_state: Any,
        global_state: Any,
        rng: np.random.Generator,
    ) -> StepOutput:
        ...

    def evaluate_personalized(self, personal_state: Any, dataset: Any) -> tuple[float, dict[str, float] | None]:
        ...

    def evaluate_global(self, global_state: Any, dataset: Any) -> tuple[float, dict[str, float] | None]:
        ...


# -----------------------------
# Dummy adapter (for smoke testing before integration)
# -----------------------------


class DummyAdapter(BasePersonalizedAdapter):
    """A minimal adapter which uses simple Gaussian parameters and toy data.

    This is only for end-to-end wiring checks. Replace with real adapter that
    wraps the user's implementation.
    """

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    # States are simple dicts containing numpy arrays
    def init_global_state(self, *, seed: int) -> Any:
        rng = np.random.default_rng(seed)
        return {"w": rng.normal(size=(self.dim,)).astype(np.float32)}

    def init_personalized_state(self, *, seed: int) -> Any:
        rng = np.random.default_rng(seed + 123)
        return {"v": rng.normal(size=(self.dim,)).astype(np.float32)}

    def global_state_to_ndarrays(self, state: Any) -> List[np.ndarray]:
        return [state["w"].copy()]

    def ndarrays_to_global_state(self, arrays: Sequence[np.ndarray]) -> Any:
        return {"w": np.array(arrays[0], copy=True)}

    def sync_personalized_with_global(self, *, personal_state: Any, global_state: Any) -> Any:
        # No-op for dummy
        return personal_state

    # Toy data: features x in R^dim, labels y = sign(w_true^T x + noise)
    def get_default_train_data(self) -> Any:
        return self._make_dataset(num=256, seed=0)

    def get_default_test_data(self) -> Any:
        return self._make_dataset(num=128, seed=1)

    @staticmethod
    def num_examples(dataset: Any) -> int:
        return int(dataset[0].shape[0])

    def minibatch_iterator(self, dataset: Any, *, batch_size: int, rng: np.random.Generator) -> Iterator[Any]:
        x, y = dataset
        n = x.shape[0]
        idx = np.arange(n)
        rng.shuffle(idx)
        for start in range(0, n, batch_size):
            sel = idx[start : min(start + batch_size, n)]
            yield (x[sel], y[sel])

    # Dummy steps simply move parameters slightly towards reducing a toy loss
    def step_personalized(
        self,
        *,
        personal_state: Any,
        global_state: Any,
        batch: Any,
        num_mc_samples: int,
        kl_weight: float,
        dataset_size: int,
        rng: np.random.Generator,
    ) -> BasePersonalizedAdapter.StepOutput:
        v = personal_state["v"]
        x, y = batch
        # Toy gradient: move v towards minimizing squared error to y for projection x@v
        pred = (x @ v)
        grad = (2.0 / max(1, x.shape[0])) * (x.T @ (pred - y))
        # KL towards global w as L2 difference
        grad += kl_weight * (v - global_state["w"])
        new_v = v - 0.05 * grad.astype(np.float32)
        loss = float(((pred - y) ** 2).mean()) + float(kl_weight * ((v - global_state["w"]) ** 2).mean())
        return BasePersonalizedAdapter.StepOutput(updated_state={"v": new_v}, metrics={"loss_personal": loss})

    def step_localized_global(
        self,
        *,
        personal_state: Any,
        global_state: Any,
        rng: np.random.Generator,
    ) -> BasePersonalizedAdapter.StepOutput:
        # Move w towards v to reduce KL (here L2 proxy)
        v = personal_state["v"]
        w = global_state["w"]
        new_w = w - 0.05 * (w - v)
        return BasePersonalizedAdapter.StepOutput(updated_state={"w": new_w}, metrics={"kl": float(((v - w) ** 2).mean())})

    def evaluate_personalized(self, personal_state: Any, dataset: Any) -> tuple[float, dict[str, float] | None]:
        x, y = dataset
        v = personal_state["v"]
        pred = x @ v
        loss = float(((pred - y) ** 2).mean())
        return loss, {"mse": loss}

    def evaluate_global(self, global_state: Any, dataset: Any) -> tuple[float, dict[str, float] | None]:
        x, y = dataset
        w = global_state["w"]
        pred = x @ w
        loss = float(((pred - y) ** 2).mean())
        return loss, {"mse": loss}

    # Utilities
    def _make_dataset(self, num: int, seed: int) -> Any:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(num, self.dim)).astype(np.float32)
        w_true = rng.normal(size=(self.dim,)).astype(np.float32)
        y = (x @ w_true + 0.1 * rng.normal(size=(num,))).astype(np.float32)
        return x, y

