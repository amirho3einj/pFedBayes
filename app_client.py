import argparse

import flwr as fl

from personalized_client import ClientRunConfig, PersonalizedClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower Client for Personalized Variational FL")
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080", help="Server address")
    parser.add_argument("--local-steps", type=int, default=10, help="R: alternating local steps per round")
    parser.add_argument("--batch-size", type=int, default=32, help="b: minibatch size")
    parser.add_argument("--mc-samples", type=int, default=1, help="K: Monte Carlo samples per minibatch")
    parser.add_argument("--kl-weight", type=float, default=1.0, help="zeta: KL weight")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--client-id", type=str, default=None, help="Optional client id for logging")
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
    client = PersonalizedClient(config=cfg, client_id=args.client_id)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()

