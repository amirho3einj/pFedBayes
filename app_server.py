import argparse

import flwr as fl

from strategy_personalized import PersonalizedStrategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower Server for Personalized Variational FL")
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080", help="gRPC server address")
    parser.add_argument("--rounds", type=int, default=3, help="Number of FL rounds")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of clients sampled per round (S). If omitted, uses fraction_fit.")
    parser.add_argument("--fraction-fit", type=float, default=1.0, help="Fraction of clients used during training")
    parser.add_argument("--min-fit-clients", type=int, default=1, help="Min number of clients for fit")
    parser.add_argument("--min-available-clients", type=int, default=1, help="Min number of available clients")
    parser.add_argument("--unweighted-mean", action="store_true", help="Use unweighted mean aggregation (default). If false, weight by num_examples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    strategy = PersonalizedStrategy(
        sample_size=args.sample_size,
        weight_by_num_examples=(not args.unweighted_mean),
        fraction_fit=args.fraction_fit,
        min_fit_clients=args.min_fit_clients,
        min_available_clients=args.min_available_clients,
    )

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()

