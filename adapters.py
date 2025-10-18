from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List, Optional, Protocol, Sequence

import numpy as np
import math
from typing import Tuple

# Optional heavy deps are imported lazily inside the adapter to keep
# DummyAdapter usable without torch installed.


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


# -----------------------------
# Adapter wrapping the existing pFedBayes implementation
# -----------------------------


class PfedBayesAdapter(BasePersonalizedAdapter):
    """Adapter integrating the provided pFedBayes plain-Python/Torch code.

    - Keeps two models on the client: `personal_model` (q_v) and `global_model` (w)
    - Alternating steps: personalized step reduces Omega^i, localized-global step reduces KL(q_v || w)
    - Serialization exposes the global posterior parameters (mus, rhos) to Flower as ndarrays
    - Respects minibatch size b, MC samples K (the original code uses K=1), and loss scaling n/b
    """

    def __init__(
        self,
        *,
        dataset: str = "Mnist",
        client_index: int = 0,
        device: str | None = None,
        input_dim: int = 28 * 28,
        hidden_dim: int = 100,
        output_dim: int = 10,
        weight_scale: float = 0.1,
        rho_offset: float = -3.0,
        zeta: float = 10.0,
    ) -> None:
        # Lazy imports (torch-heavy) to avoid forcing torch for Dummy users
        import torch  # type: ignore
        from FLAlgorithms.trainmodel.models import pBNN  # type: ignore
        from utils.model_utils import read_data, read_user_data  # type: ignore

        self.dataset_name = dataset
        self.client_index = client_index
        self._torch = torch
        self._pBNN = pBNN
        self.device = torch.device(device if device is not None else "cpu")

        # Build model templates
        self.model_kwargs = dict(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            device=self.device,
            weight_scale=weight_scale,
            rho_offset=rho_offset,
            zeta=zeta,
        )

        # Load per-client data split
        data = read_data(dataset)
        _id, train, test = read_user_data(client_index, data, dataset, device=self.device)
        self._train_data = train
        self._test_data = test

        # Optimizers are created on first init of states
        self._opt_personal = None
        self._opt_global = None

    # ------ States and parameterization ------
    def _new_model(self, *, seed: int) -> Any:
        torch = self._torch
        torch.manual_seed(seed)
        return self._pBNN(**self.model_kwargs)

    def init_global_state(self, *, seed: int) -> Any:
        model = self._new_model(seed=seed)
        # Create optimizer for global model
        self._opt_global = self._torch.optim.Adam(model.parameters(), lr=1e-3)
        return model

    def init_personalized_state(self, *, seed: int) -> Any:
        model = self._new_model(seed=seed + 123)
        # Create optimizer for personal model
        self._opt_personal = self._torch.optim.Adam(model.parameters(), lr=1e-3)
        return model

    def _flatten_posterior(self, model: Any) -> List[np.ndarray]:
        # Expose [mu0, rho0, mu1, rho1, ...] as numpy arrays
        arrays: List[np.ndarray] = []
        for mu, rho in zip(model.mus, model.rhos):
            arrays.append(mu.detach().cpu().numpy().astype(np.float32))
            arrays.append(rho.detach().cpu().numpy().astype(np.float32))
        return arrays

    def global_state_to_ndarrays(self, state: Any) -> List[np.ndarray]:
        return self._flatten_posterior(state)

    def ndarrays_to_global_state(self, arrays: Sequence[np.ndarray]) -> Any:
        # Rebuild model and load mus/rhos
        model = self._new_model(seed=0)
        # Assign parameters
        assert len(arrays) == 2 * len(model.mus), "Mismatched number of arrays for mus/rhos"
        for i, (mu, rho) in enumerate(zip(model.mus, model.rhos)):
            mu_arr = self._torch.tensor(arrays[2 * i], device=self.device, dtype=mu.dtype)
            rho_arr = self._torch.tensor(arrays[2 * i + 1], device=self.device, dtype=rho.dtype)
            with self._torch.no_grad():
                mu.copy_(mu_arr)
                rho.copy_(rho_arr)
        # (Re-)create optimizer bound to this state if needed
        if self._opt_global is None:
            self._opt_global = self._torch.optim.Adam(model.parameters(), lr=1e-3)
        return model

    def sync_personalized_with_global(self, *, personal_state: Any, global_state: Any) -> Any:
        # Reset personalized model to the current global posterior (server broadcast)
        torch = self._torch
        with torch.no_grad():
            for p_mu, p_rho, g_mu, g_rho in zip(personal_state.mus, personal_state.rhos, global_state.mus, global_state.rhos):
                p_mu.copy_(g_mu)
                p_rho.copy_(g_rho)
        return personal_state

    # ------ Data utilities ------
    def get_default_train_data(self) -> Any:
        return self._train_data

    def get_default_test_data(self) -> Any:
        return self._test_data

    @staticmethod
    def num_examples(dataset: Any) -> int:
        return int(len(dataset))

    def minibatch_iterator(self, dataset: Any, *, batch_size: int, rng: np.random.Generator) -> Iterator[Any]:
        # dataset is a list of (x, y) torch tensors
        indices = np.arange(len(dataset))
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            sel = indices[start : min(start + batch_size, len(indices))]
            xs = [dataset[i][0] for i in sel]
            ys = [dataset[i][1] for i in sel]
            # Stack into mini-batch tensors
            x_batch = self._torch.stack(xs, dim=0)
            y_batch = self._torch.stack(ys, dim=0)
            yield (x_batch, y_batch)

    # ------ Learning steps and evaluation ------
    @dataclass
    class StepOutput:
        updated_state: Any
        metrics: Optional[dict[str, float]] = None

    def _one_hot(self, y: Any, num_classes: int) -> Any:
        F = self._torch.nn.functional
        return F.one_hot(y.view(-1), num_classes=num_classes).to(dtype=self._torch.float32)

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
        torch = self._torch
        F = torch.nn.functional
        x, y = batch
        # Flatten to vectors as in original code
        x = x.view(x.size(0), -1).to(self.device)
        y = y.view(y.size(0), -1).to(self.device)
        label_one_hot = F.one_hot(y.view(-1), num_classes=self.model_kwargs["output_dim"]).to(dtype=torch.float32)
        # Ensure personal_state knows current zeta/kl weight if needed
        personal_state.zeta = torch.tensor(kl_weight, device=self.device)

        # Estimate number of batches for scaling (n/b)
        batch_size = x.size(0)
        num_batches = max(1, math.ceil(dataset_size / max(1, batch_size)))

        # Average MC samples (original code effectively uses K=1)
        loss_accum = 0.0
        for _ in range(max(1, num_mc_samples)):
            eps = personal_state.sample_epsilons(personal_state.layer_param_shapes)
            layer_params = personal_state.transform_gaussian_samples(personal_state.mus, personal_state.rhos, eps)
            logits = personal_state.net(x, layer_params)
            loss_k = personal_state.combined_loss_personal(
                logits,
                label_one_hot,
                layer_params,
                personal_state.mus,
                personal_state.sigmas,
                [t.clone().detach() for t in global_state.mus],
                [t.clone().detach() for t in global_state.sigmas],
                num_batches,
            )
            loss_accum = loss_accum + loss_k
        loss = loss_accum / float(max(1, num_mc_samples))
        # Rescale to match n/b scaling by multiplying by estimated num_batches
        loss = loss * float(num_batches)

        assert self._opt_personal is not None, "Personal optimizer not initialized"
        self._opt_personal.zero_grad()
        loss.backward()
        self._opt_personal.step()

        return BasePersonalizedAdapter.StepOutput(
            updated_state=personal_state,
            metrics={"loss_personal": float(loss.detach().cpu().item())},
        )

    def step_localized_global(
        self,
        *,
        personal_state: Any,
        global_state: Any,
        rng: np.random.Generator,
    ) -> BasePersonalizedAdapter.StepOutput:
        torch = self._torch
        # Ensure sigmas are consistent with current rhos
        personal_sigmas = personal_state.transform_rhos(personal_state.rhos)
        global_sigmas = global_state.transform_rhos(global_state.rhos)

        # Temporarily set zeta=1.0 to match Omega_w^i = KL(q_v || w)
        zeta_backup = getattr(global_state, "zeta", torch.tensor(1.0, device=self.device))
        global_state.zeta = torch.tensor(1.0, device=self.device)

        # Minimize KL(q_v || w) without extra scaling
        loss = global_state.combined_loss_local(
            list(global_state.mus),
            [t.clone().detach() for t in personal_state.mus],
            [t.clone().detach() for t in personal_sigmas],
            global_state.mus,
            global_sigmas,
            1,
        )

        assert self._opt_global is not None, "Global optimizer not initialized"
        self._opt_global.zero_grad()
        loss.backward()
        self._opt_global.step()

        # Restore original zeta
        global_state.zeta = zeta_backup

        # Metric: KL magnitude proxy (L2 between means) for logging
        kl_proxy = 0.0
        with torch.no_grad():
            for vm, wm in zip(personal_state.mus, global_state.mus):
                kl_proxy += float(((vm - wm) ** 2).mean().cpu().item())

        return BasePersonalizedAdapter.StepOutput(updated_state=global_state, metrics={"kl": kl_proxy})

    # Evaluation utilities
    def _evaluate_model(self, model: Any, dataset: Any) -> Tuple[float, dict[str, float] | None]:
        torch = self._torch
        F = torch.nn.functional
        total_loss = 0.0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            # Use a single MC sample per batch (original semantics)
            for x, y in dataset:
                x = x.view(x.size(0), -1).to(self.device)
                y = y.view(y.size(0), -1).to(self.device)
                eps = model.sample_epsilons(model.layer_param_shapes)
                layer_params = model.transform_gaussian_samples(model.mus, model.rhos, eps)
                logits = model.net(x, layer_params)
                ce = F.cross_entropy(logits, y.view(-1), reduction="sum")
                total_loss += float(ce.detach().cpu().item())
                pred = logits.argmax(dim=1)
                correct += int((pred == y.view(-1)).sum().cpu().item())
                total += int(y.numel())
        model.train()
        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)
        return avg_loss, {"accuracy": acc}

    def evaluate_personalized(self, personal_state: Any, dataset: Any) -> tuple[float, dict[str, float] | None]:
        return self._evaluate_model(personal_state, dataset)

    def evaluate_global(self, global_state: Any, dataset: Any) -> tuple[float, dict[str, float] | None]:
        return self._evaluate_model(global_state, dataset)
