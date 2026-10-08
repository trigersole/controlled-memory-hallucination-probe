from __future__ import annotations

import argparse
import json
import platform
import sys

import torch

from .benchmark import collect_benchmark, evaluate_benchmark, intervention_report
from .collect import collect
from .config import initialize_output, load_config, output_dir
from .probe import train_all, train_one
from .synthetic import generate
from .summarize import summarize_results
from .train_adapter import train
from .versioning import artifact_metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Controlled-memory PEFT probe experiment")
    parser.add_argument("--config", default="configs/pilot.yaml", help="YAML experiment configuration")
    parser.add_argument("--force", action="store_true", help="Recompute completed outputs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate", help="Validate configuration and print the runtime environment")
    subparsers.add_parser("generate-data", help="Generate deterministic fictional facts")

    adapter = subparsers.add_parser("train-adapter", help="Train or resume one LoRA adapter")
    adapter.add_argument("--adapter", choices=("a", "b"), required=True)

    collection = subparsers.add_parser("collect-synthetic", help="Collect one source in atomic shards")
    collection.add_argument("--source", choices=("adapter_a", "adapter_b", "base"), required=True)

    subparsers.add_parser("intervention-report", help="Measure exposed versus withheld accuracy")

    probe = subparsers.add_parser("train-probe", help="Train or resume one probe")
    probe.add_argument("--feature-mode", choices=("base_replay", "on_policy"), required=True)
    probe.add_argument(
        "--variant", choices=("correctness", "shuffled_exposure", "genuine_exposure"), required=True
    )
    probe.add_argument("--seed", type=int, required=True)
    subparsers.add_parser("train-probes", help="Train every configured feature/variant/seed probe")

    benchmark = subparsers.add_parser("collect-benchmark", help="Generate and featurize a benchmark")
    benchmark.add_argument("--benchmark", choices=("trivia_qa", "truthful_qa"), required=True)
    evaluation = subparsers.add_parser("evaluate", help="Evaluate every trained probe on a benchmark")
    evaluation.add_argument("--benchmark", choices=("trivia_qa", "truthful_qa"), required=True)
    subparsers.add_parser("summarize", help="Write a compact Markdown analysis of completed results")
    return parser


def _validate(config: dict) -> None:
    required = ("experiment", "model", "data", "lora", "collection", "probe", "benchmarks")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")
    information = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "model": config["model"]["name_or_path"],
        "output_dir": str(output_dir(config)),
        **artifact_metadata(),
    }
    print(json.dumps(information, indent=2))


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    initialize_output(config)
    if args.command == "validate":
        _validate(config)
    elif args.command == "generate-data":
        print(generate(config, force=args.force))
    elif args.command == "train-adapter":
        print(train(config, args.adapter, force=args.force))
    elif args.command == "collect-synthetic":
        print(collect(config, args.source, force=args.force))
    elif args.command == "intervention-report":
        print(intervention_report(config))
    elif args.command == "train-probe":
        print(train_one(config, args.feature_mode, args.variant, args.seed, force=args.force))
    elif args.command == "train-probes":
        for path in train_all(config, force=args.force):
            print(path)
    elif args.command == "collect-benchmark":
        print(collect_benchmark(config, args.benchmark, force=args.force))
    elif args.command == "evaluate":
        print(evaluate_benchmark(config, args.benchmark, force=args.force))
    elif args.command == "summarize":
        print(summarize_results(config))


if __name__ == "__main__":
    main()
