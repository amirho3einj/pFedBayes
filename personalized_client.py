from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import flwr as fl

from adapters import BasePersonalizedAdapter, DummyAdapter


@dataclass
class ClientRunConfig:
    # Local alternating steps per round
    local_steps: int = 1  # R
    minibatch_size: int = 32  # b
    num_mc_samples: int = 1  # K
    kl_weight: float = 1.0  # zeta
    # Learning rates or optimizer configs can be adapter-internal; expose here if needed
    seed: int = 0
    # Metrics verbosity options
    report_kl: bool = True
    report_nll: bool = True


class PersonalizedClient(fl.client.NumPyClient):
    """Flower client implementing the alternating personalized/localized-global routine.

    This client defers actual math to an Adapter which wraps the user's existing
    model, losses, sampling, and optimizers. This keeps integration minimal.
    """

    def __init__(
        self,
        *,
        adapter: Optional[BasePersonalizedAdapter] = None,
        train_data: Any | None = None,
        test_data: Any | None = None,
        config: Optional[ClientRunConfig] = None,
        client_id: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.adapter: BasePersonalizedAdapter = adapter or DummyAdapter()
        self.cfg: ClientRunConfig = config or ClientRunConfig()
        self.train_data = train_data if train_data is not None else self.adapter.get_default_train_data()
        self.test_data = test_data if test_data is not None else self.adapter.get_default_test_data()
        self.client_id = client_id

        # Client-local states
        self.personal_state = self.adapter.init_personalized_state(seed=self.cfg.seed)
        self.localized_global_state = self.adapter.init_global_state(seed=self.cfg.seed)

        # Cache dataset size
        self.num_examples_train = self.adapter.num_examples(self.train_data)
        self.num_examples_test = self.adapter.num_examples(self.test_data)

    # Flower will use this when no strategy-level init is provided
    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List[np.ndarray]:  # type: ignore[override]
        return self.adapter.global_state_to_ndarrays(self.localized_global_state)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, fl.common.Scalar],
    ) -> Tuple[List[np.ndarray], int, Dict[str, fl.common.Scalar]]:  # type: ignore[override]
        # Deserialize global parameters into the client's localized-global copy
        self.localized_global_state = self.adapter.ndarrays_to_global_state(parameters)
        # Optionally (re-)condition personalized state on new global
        self.personal_state = self.adapter.sync_personalized_with_global(
            personal_state=self.personal_state,
            global_state=self.localized_global_state,
        )

        R = int(config.get("local_steps", self.cfg.local_steps))
        b = int(config.get("minibatch_size", self.cfg.minibatch_size))
        K = int(config.get("num_mc_samples", self.cfg.num_mc_samples))
        zeta = float(config.get("kl_weight", self.cfg.kl_weight))
        seed = int(config.get("seed", self.cfg.seed))

        rng = np.random.default_rng(seed + (0 if self.client_id is None else hash(self.client_id) % 10000))

        # Alternating local routine
        metrics_accum: Dict[str, float] = {"loss_personal": 0.0, "kl": 0.0, "nll": 0.0}
        steps_done = 0
        data_iter = self.adapter.minibatch_iterator(self.train_data, batch_size=b, rng=rng)
        for _ in range(R):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = self.adapter.minibatch_iterator(self.train_data, batch_size=b, rng=rng)
                batch = next(data_iter)

            # 1) Personalized step on v^i
            personal_out = self.adapter.step_personalized(
                personal_state=self.personal_state,
                global_state=self.localized_global_state,
                batch=batch,
                num_mc_samples=K,
                kl_weight=zeta,
                dataset_size=self.num_examples_train,
                rng=rng,
            )
            self.personal_state = personal_out.updated_state

            # 2) Localized-global step on ~w^i (towards current global posterior)
            global_out = self.adapter.step_localized_global(
                personal_state=self.personal_state,
                global_state=self.localized_global_state,
                rng=rng,
            )
            self.localized_global_state = global_out.updated_state

            # Accumulate metrics if provided
            if personal_out.metrics is not None:
                for k, v in personal_out.metrics.items():
                    metrics_accum[k] = metrics_accum.get(k, 0.0) + float(v)
            steps_done += 1

        # Average metrics over steps
        fit_metrics: Dict[str, fl.common.Scalar] = {}
        if steps_done > 0:
            for k, v in metrics_accum.items():
                fit_metrics[k] = v / steps_done
        fit_metrics["client_id"] = str(self.client_id) if self.client_id is not None else "unknown"

        # Upload ONLY localized-global ~w^i
        updated_params = self.adapter.global_state_to_ndarrays(self.localized_global_state)
        return updated_params, int(self.num_examples_train), fit_metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, fl.common.Scalar],
    ) -> Tuple[float, int, Dict[str, fl.common.Scalar]]:  # type: ignore[override]
        # Option A (preferred): evaluate personalized model
        eval_mode = str(config.get("eval_mode", "personalized"))

        # Ensure local global copy is up to date (in case server sent new params for eval)
        self.localized_global_state = self.adapter.ndarrays_to_global_state(parameters)
        self.personal_state = self.adapter.sync_personalized_with_global(
            personal_state=self.personal_state,
            global_state=self.localized_global_state,
        )

        if eval_mode == "global":
            loss, metrics = self.adapter.evaluate_global(self.localized_global_state, self.test_data)
        else:
            loss, metrics = self.adapter.evaluate_personalized(self.personal_state, self.test_data)

        return float(loss), int(self.num_examples_test), metrics or {}

