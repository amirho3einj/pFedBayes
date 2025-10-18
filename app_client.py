import argparse

import flwr as fl

from personalized_client import ClientRunConfig, PersonalizedClient
from adapters import PfedBayesAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower Client for Personalized Variational FL")
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080", help="Server address")
    parser.add_argument("--local-steps", type=int, default=10, help="R: alternating local steps per round")
    parser.add_argument("--batch-size", type=int, default=32, help="b: minibatch size")
    parser.add_argument("--mc-samples", type=int, default=1, help="K: Monte Carlo samples per minibatch")
    parser.add_argument("--kl-weight", type=float, default=1.0, help="zeta: KL weight")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--client-id", type=str, default=None, help="Optional client id for logging")
    parser.add_argument("--dataset", type=str, default="Mnist", help="Dataset name for adapter (e.g., Mnist, FMnist)")
    parser.add_argument("--device", type=str, default=None, help="Torch device string, e.g., 'cuda:0' or 'cpu'")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ClientRunConfig(
        local_steps=args.local_steps,
        minibatch_size=args.batch_size,
        num_mc_samples=args.mc_samples,
        kl_weight=args.kl_weight,
        seed=args.seed,
    )
    # Use the pFedBayes adapter to wrap the existing implementation
    # Map client-id to integer index for per-client data selection
    try:
        client_index = int(args.client_id) if args.client_id is not None else 0
    except ValueError:
        client_index = 0
    adapter = PfedBayesAdapter(dataset=args.dataset, client_index=client_index, device=args.device)
    client = PersonalizedClient(adapter=adapter, config=cfg, client_id=args.client_id)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()

