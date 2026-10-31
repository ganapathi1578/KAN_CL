"""Command-line interface."""
from __future__ import annotations

import argparse
import json

from kan_cl.engine.experiment import load_config, run_experiment
from kan_cl.registry import BASIS_FUNCTIONS, MODELS


def main() -> None:
    parser = argparse.ArgumentParser(prog="kan-cl", description="KAN continual-learning framework")
    parser.add_argument("--config", type=str, help="YAML or JSON experiment config")
    parser.add_argument("--output", type=str, default="outputs/run", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or cuda:N")
    parser.add_argument("--list-models", action="store_true", help="List registered model names")
    parser.add_argument("--list-bases", action="store_true", help="List registered basis names")
    args = parser.parse_args()
    if args.list_models:
        print("\n".join(MODELS.names()))
        return
    if args.list_bases:
        print("\n".join(BASIS_FUNCTIONS.names()))
        return
    if not args.config:
        parser.error("--config is required unless --list-models or --list-bases is used")
    config = load_config(args.config)
    result = run_experiment(config, args.output, args.device)
    print(json.dumps(result["metrics"], indent=2, allow_nan=True))
    print(f"Saved results and checkpoint artifacts to: {args.output}")


if __name__ == "__main__":
    main()
