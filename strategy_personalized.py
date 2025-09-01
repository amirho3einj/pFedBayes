from typing import Dict, List, Optional, Tuple

import numpy as np

import flwr as fl
from flwr.common import (
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)


class PersonalizedStrategy(fl.server.strategy.FedAvg):
    """FedAvg-like strategy with configurable client sampling and aggregation.

    Differences from FedAvg:
    - configure_fit: sample exactly `sample_size` clients per round (if available)
    - aggregate_fit: supports unweighted mean across clients (default) to match
      paper description; optionally weight by num_examples.
    """

    def __init__(
        self,
        *,
        sample_size: Optional[int] = None,
        weight_by_num_examples: bool = False,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 1,
        min_available_clients: int = 1,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=0,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.sample_size = sample_size
        self.weight_by_num_examples = weight_by_num_examples

    # Flower will call this on every round to select clients and provide FitIns
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, FitIns]]:
        # If a fixed sample_size is set, sample exactly that many (if possible)
        if self.sample_size is not None:
            num_available = client_manager.num_available()
            desired = min(self.sample_size, num_available)
            if desired < self.sample_size:
                # Not enough clients available; fall back to as many as possible
                pass
            fit_config: Dict[str, Scalar] = {"server_round": server_round}
            clients = client_manager.sample(num_clients=desired, min_num_clients=desired)
            fit_ins = FitIns(parameters, fit_config)
            return [(client, fit_ins) for client in clients]

        # Otherwise, defer to FedAvg's default behavior
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]] ,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # Convert all client Parameters to ndarrays
        client_ndarrays: List[NDArrays] = [parameters_to_ndarrays(res.parameters) for _, res in results]
        # Determine aggregation weights
        if self.weight_by_num_examples:
            weights = np.array([res.num_examples for _, res in results], dtype=np.float64)
            weights = weights / (weights.sum() + 1e-12)
        else:
            weights = None

        # Simple mean (unweighted) or weighted by num_examples
        aggregated: NDArrays = []
        num_clients = len(client_ndarrays)
        for param_idx in range(len(client_ndarrays[0])):
            stacked = np.stack([client_ndarrays[c][param_idx] for c in range(num_clients)], axis=0)
            if weights is None:
                mean_param = stacked.mean(axis=0)
            else:
                mean_param = np.tensordot(weights, stacked, axes=1)
            aggregated.append(mean_param)

        aggregated_parameters = ndarrays_to_parameters(aggregated)

        # Optionally aggregate metrics (use FedAvg's helper)
        metrics_aggregated: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn is not None:
            fit_metrics = [res.metrics for _, res in results]
            num_examples = [res.num_examples for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics, num_examples)

        return aggregated_parameters, metrics_aggregated

