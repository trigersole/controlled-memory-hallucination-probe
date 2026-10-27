from __future__ import annotations

import argparse
import json
import platform
import sys

import torch

from .adapter_invariant import analyze_adapter_invariant, load_adapter_invariant_config
from .adapter_invariant_subspace import (
    analyze_adapter_invariant_subspace,
    load_adapter_invariant_subspace_config,
)
from .adapter_transfer import evaluate_adapter_transfer
from .base_contextual_transfer import (
    analyze_popqa_base_support,
    collect_popqa_base_support,
    collect_popqa_candidates,
    load_base_contextual_transfer_config,
)
from .benchmark import collect_benchmark, evaluate_benchmark, intervention_report
from .collect import collect, configured_feature_keys
from .comparison import (
    analyze_matched_comparison,
    collect_comparison_benchmark,
    collect_comparison_synthetic,
    load_comparison_config,
)
from .confidence_gate import analyze_confidence_gate, load_confidence_gate_config
from .config import initialize_output, load_config, output_dir, validate_source_run
from .cross_model_rank_gate import (
    analyze_cross_model_rank_gate,
    load_cross_model_rank_gate_config,
)
from .cross_model_support import (
    analyze_cross_model_support,
    collect_cross_model_support,
    load_cross_model_support_config,
)
from .geometry import (
    analyze_geometry,
    collect_benchmark_geometry,
    collect_synthetic_geometry,
    load_geometry_config,
)
from .memory_subspace import analyze_memory_subspace, load_memory_subspace_config
from .probe import train_all, train_one
from .summarize import summarize_results
from .synthetic import generate
from .train_adapter import train
from .training_shift import (
    analyze_training_shift,
    collect_training_shift,
    load_training_shift_config,
)
from .versioning import artifact_metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Controlled-memory PEFT probe experiment")
    parser.add_argument("--config", default="configs/pilot.yaml", help="YAML experiment configuration")
    parser.add_argument(
        "--geometry-config",
        default="configs/geometry.yaml",
        help="Separate hidden-space geometry settings",
    )
    parser.add_argument(
        "--comparison-config",
        default="configs/haloscope_comparison.yaml",
        help="Matched HaloScope-style comparison settings",
    )
    parser.add_argument(
        "--memory-subspace-config",
        default="configs/memory_subspace.yaml",
        help="Controlled-memory cosine/subspace analysis settings",
    )
    parser.add_argument(
        "--adapter-invariant-config",
        default="configs/adapter_invariant.yaml",
        help="Adapter-invariant correctness-pattern settings",
    )
    parser.add_argument(
        "--adapter-invariant-subspace-config",
        default="configs/adapter_invariant_subspace.yaml",
        help="Adapter-invariant multidimensional correctness-subspace settings",
    )
    parser.add_argument(
        "--confidence-gate-config",
        default="configs/confidence_gate.yaml",
        help="Validation-selected confidence-gate settings",
    )
    parser.add_argument(
        "--cross-model-support-config",
        default="configs/cross_model_support.yaml",
        help="Cross-model normalized answer-support settings",
    )
    parser.add_argument(
        "--cross-model-rank-gate-config",
        default="configs/cross_model_rank_gate.yaml",
        help="Fixed-coverage rank-normalized support-gate settings",
    )
    parser.add_argument(
        "--base-contextual-transfer-config",
        default="configs/popqa_base_contextual_support.yaml",
        help="Isolated base-only contextual-support transfer settings",
    )
    parser.add_argument(
        "--training-shift-config",
        default="configs/training_shift.yaml",
        help="Before/after LoRA factual-subspace settings",
    )
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
    probe.add_argument(
        "--feature-mode",
        required=True,
        help="Configured feature key, e.g. base_replay__prompt_last",
    )
    probe.add_argument(
        "--variant", choices=("correctness", "shuffled_exposure", "genuine_exposure"), required=True
    )
    probe.add_argument("--seed", type=int, required=True)
    subparsers.add_parser("train-probes", help="Train every configured feature/variant/seed probe")
    subparsers.add_parser(
        "adapter-transfer",
        help="Train on each adapter separately and test transfer to the other adapter",
    )

    benchmark = subparsers.add_parser("collect-benchmark", help="Generate and featurize a benchmark")
    benchmark.add_argument("--benchmark", choices=("trivia_qa", "truthful_qa"), required=True)
    evaluation = subparsers.add_parser("evaluate", help="Evaluate every trained probe on a benchmark")
    evaluation.add_argument("--benchmark", choices=("trivia_qa", "truthful_qa"), required=True)
    subparsers.add_parser("summarize", help="Write a compact Markdown analysis of completed results")
    geometry_synthetic = subparsers.add_parser(
        "collect-geometry-synthetic", help="Collect prompt-only multi-layer adapter features"
    )
    geometry_synthetic.add_argument("--source", choices=("adapter_a", "adapter_b"), required=True)
    geometry_benchmark = subparsers.add_parser(
        "collect-geometry-benchmark", help="Collect prompt-only multi-layer benchmark features"
    )
    geometry_benchmark.add_argument(
        "--benchmark", choices=("trivia_qa", "truthful_qa"), required=True
    )
    subparsers.add_parser(
        "analyze-geometry", help="Fit memory subspaces and evaluate geometric transfer"
    )
    comparison_synthetic = subparsers.add_parser(
        "collect-comparison-synthetic",
        help="Collect answer-conditioned paired adapter features",
    )
    comparison_synthetic.add_argument(
        "--source", choices=("adapter_a", "adapter_b"), required=True
    )
    comparison_benchmark = subparsers.add_parser(
        "collect-comparison-benchmark",
        help="Replay saved benchmark answers for matched multi-layer features",
    )
    comparison_benchmark.add_argument(
        "--benchmark", choices=("trivia_qa", "truthful_qa"), required=True
    )
    subparsers.add_parser(
        "analyze-comparison",
        help="Evaluate controlled memory against matched HaloScope-style projection",
    )
    subparsers.add_parser(
        "analyze-memory-subspace",
        help="Evaluate cosine, subspace, and answer-trajectory memory scores",
    )
    subparsers.add_parser(
        "analyze-adapter-invariant",
        help="Find a correctness pattern shared by A and B after removing adapter identity",
    )
    subparsers.add_parser(
        "analyze-adapter-invariant-subspace",
        help="Find a rotation-invariant correctness subspace shared by A and B",
    )
    subparsers.add_parser(
        "analyze-confidence-gate",
        help="Select a confidence cutoff on synthetic validation and freeze it for evaluation",
    )
    cross_model_collection = subparsers.add_parser(
        "collect-cross-model-support",
        help="Replay fixed candidate answers through base, LoRA A, or LoRA B",
    )
    cross_model_collection.add_argument(
        "--evaluator", choices=("base", "adapter_a", "adapter_b"), required=True
    )
    subparsers.add_parser(
        "analyze-cross-model-support",
        help="Select normalized support features on synthetic validation and evaluate transfer",
    )
    subparsers.add_parser(
        "analyze-cross-model-rank-gate",
        help="Evaluate fixed-coverage rank gates without natural-label calibration",
    )
    subparsers.add_parser(
        "collect-popqa-candidates",
        help="Generate base-model answers for the isolated PopQA transfer test",
    )
    subparsers.add_parser(
        "collect-popqa-base-support",
        help="Score PopQA answers against hard relation-matched controls",
    )
    subparsers.add_parser(
        "analyze-popqa-base-support",
        help="Evaluate the label-free base-only contextual-support ranking on PopQA",
    )
    training_shift = subparsers.add_parser(
        "collect-training-shift",
        help="Collect matched base and LoRA factual-prompt activations",
    )
    training_shift.add_argument(
        "--source", choices=("base", "adapter_a", "adapter_b"), required=True
    )
    subparsers.add_parser(
        "analyze-training-shift",
        help="Compare principal activation and LoRA-minus-base update subspaces",
    )
    for command, help_text in (
        ("validate-factual-shift", "Validate the isolated factual LoRA experiment"),
        ("prepare-factual-shift", "Create independent TriviaQA experiment splits"),
        ("prepare-factual-shift-negatives", "Sample incorrect-answer control targets"),
        ("collect-factual-shift-candidates", "Save fixed base-model answer candidates"),
        ("analyze-factual-shift", "Test whether LoRA update directions predict correctness"),
        ("audit-factual-shift", "Check every saved factual feature shard for corruption"),
        ("repair-factual-shift", "Repair damaged feature shards and finish analysis"),
        ("within-question-factual-shift", "Evaluate frozen answer scores within each question"),
        ("final-checks-factual-shift", "Compare frozen transfer and incorrect-control answer rankings"),
        ("run-factual-shift", "Run or resume every factual LoRA experiment stage"),
        ("run-study-factual-shift", "Run an isolated confirmation seed and its final diagnostics"),
    ):
        subparsers.add_parser(command, help=help_text)
    factual_train = subparsers.add_parser("train-factual-shift", help="Train one factual/control LoRA")
    factual_train.add_argument("--adapter", choices=("correct_a", "correct_b", "incorrect"), required=True)
    factual_collect = subparsers.add_parser("collect-factual-shift", help="Replay fixed text through one model")
    factual_collect.add_argument("--source", choices=("base", "correct_a", "correct_b", "incorrect"), required=True)
    return parser


def _validate(config: dict) -> None:
    required = ("experiment", "model", "data", "lora", "collection", "probe", "benchmarks")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")
    feature_names = configured_feature_keys(config)
    source_root = validate_source_run(config)
    information = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "model": config["model"]["name_or_path"],
        "output_dir": str(output_dir(config)),
        "source_output_dir": str(source_root),
        "feature_keys": feature_names,
        **artifact_metadata(),
    }
    print(json.dumps(information, indent=2))


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    if "factual-shift" in args.command:
        from . import factual_shift as factual

        factual.validate(config)
        initialize_output(config)
        if args.command == "validate-factual-shift":
            print(json.dumps({"valid": True, "cuda_available": torch.cuda.is_available(),
                              "output_dir": str(factual.root(config))}, indent=2))
        elif args.command == "prepare-factual-shift":
            print(factual.prepare(config, args.force))
        elif args.command == "prepare-factual-shift-negatives":
            print(factual.prepare_negatives(config, args.force))
        elif args.command == "train-factual-shift":
            print(factual.train(config, args.adapter, args.force))
        elif args.command == "collect-factual-shift-candidates":
            print(factual.collect_candidates(config, args.force))
        elif args.command == "collect-factual-shift":
            print(factual.collect_features(config, args.source, args.force))
        elif args.command == "analyze-factual-shift":
            print(factual.analyze(config, args.force))
        elif args.command == "final-checks-factual-shift":
            from .factual_final import analyze_final_checks

            print(analyze_final_checks(config, args.force))
        elif args.command == "within-question-factual-shift":
            from .within_question import analyze_saved_predictions

            print(analyze_saved_predictions(
                factual.root(config) / "results", int(config["factual_shift"]["bootstrap_samples"]),
                int(config["experiment"]["seed"]), args.force,
            ))
        elif args.command == "audit-factual-shift":
            report = factual.audit_features(config)
            print(json.dumps(report, indent=2))
            if not report["healthy"]:
                raise SystemExit(2)
        elif args.command == "repair-factual-shift":
            if args.force:
                raise ValueError("Repair preserves valid shards; omit --force")
            print(factual.repair_features(config), flush=True)
            print(factual.analyze(config), flush=True)
        else:
            print(factual.prepare(config, args.force), flush=True)
            print(factual.prepare_negatives(config, args.force), flush=True)
            for adapter in factual.ADAPTERS:
                print(factual.train(config, adapter, args.force), flush=True)
                factual._release()
            print(factual.collect_candidates(config, args.force), flush=True)
            for source in factual.SOURCES:
                print(factual.collect_features(config, source, args.force), flush=True)
            print(factual.analyze(config, args.force), flush=True)
            if args.command == "run-study-factual-shift":
                from .factual_final import analyze_final_checks

                print(analyze_final_checks(config, args.force), flush=True)
        return
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
    elif args.command == "adapter-transfer":
        print(evaluate_adapter_transfer(config, force=args.force))
    elif args.command == "collect-benchmark":
        print(collect_benchmark(config, args.benchmark, force=args.force))
    elif args.command == "evaluate":
        print(evaluate_benchmark(config, args.benchmark, force=args.force))
    elif args.command == "summarize":
        print(summarize_results(config))
    elif args.command == "collect-geometry-synthetic":
        settings = load_geometry_config(args.geometry_config)
        print(collect_synthetic_geometry(config, settings, args.source, force=args.force))
    elif args.command == "collect-geometry-benchmark":
        settings = load_geometry_config(args.geometry_config)
        print(collect_benchmark_geometry(config, settings, args.benchmark, force=args.force))
    elif args.command == "analyze-geometry":
        settings = load_geometry_config(args.geometry_config)
        print(analyze_geometry(config, settings, force=args.force))
    elif args.command == "collect-comparison-synthetic":
        geometry_settings = load_geometry_config(args.geometry_config)
        comparison_settings = load_comparison_config(args.comparison_config)
        print(
            collect_comparison_synthetic(
                config,
                geometry_settings,
                comparison_settings,
                args.source,
                force=args.force,
            )
        )
    elif args.command == "collect-comparison-benchmark":
        geometry_settings = load_geometry_config(args.geometry_config)
        comparison_settings = load_comparison_config(args.comparison_config)
        print(
            collect_comparison_benchmark(
                config,
                geometry_settings,
                comparison_settings,
                args.benchmark,
                force=args.force,
            )
        )
    elif args.command == "analyze-comparison":
        geometry_settings = load_geometry_config(args.geometry_config)
        comparison_settings = load_comparison_config(args.comparison_config)
        print(
            analyze_matched_comparison(
                config,
                geometry_settings,
                comparison_settings,
                force=args.force,
            )
        )
    elif args.command == "analyze-memory-subspace":
        settings = load_memory_subspace_config(args.memory_subspace_config)
        print(analyze_memory_subspace(config, settings, force=args.force))
    elif args.command == "analyze-adapter-invariant":
        settings = load_adapter_invariant_config(args.adapter_invariant_config)
        print(analyze_adapter_invariant(config, settings, force=args.force))
    elif args.command == "analyze-adapter-invariant-subspace":
        settings = load_adapter_invariant_subspace_config(
            args.adapter_invariant_subspace_config
        )
        print(analyze_adapter_invariant_subspace(config, settings, force=args.force))
    elif args.command == "analyze-confidence-gate":
        settings = load_confidence_gate_config(args.confidence_gate_config)
        print(analyze_confidence_gate(config, settings, force=args.force))
    elif args.command == "collect-cross-model-support":
        settings = load_cross_model_support_config(args.cross_model_support_config)
        print(
            collect_cross_model_support(
                config, settings, args.evaluator, force=args.force
            )
        )
    elif args.command == "analyze-cross-model-support":
        settings = load_cross_model_support_config(args.cross_model_support_config)
        print(analyze_cross_model_support(config, settings, force=args.force))
    elif args.command == "analyze-cross-model-rank-gate":
        support_settings = load_cross_model_support_config(
            args.cross_model_support_config
        )
        rank_settings = load_cross_model_rank_gate_config(
            args.cross_model_rank_gate_config
        )
        print(
            analyze_cross_model_rank_gate(
                config,
                support_settings,
                rank_settings,
                force=args.force,
            )
        )
    elif args.command == "collect-popqa-candidates":
        settings = load_base_contextual_transfer_config(
            args.base_contextual_transfer_config
        )
        print(collect_popqa_candidates(config, settings, force=args.force))
    elif args.command == "collect-popqa-base-support":
        settings = load_base_contextual_transfer_config(
            args.base_contextual_transfer_config
        )
        print(collect_popqa_base_support(config, settings, force=args.force))
    elif args.command == "analyze-popqa-base-support":
        settings = load_base_contextual_transfer_config(
            args.base_contextual_transfer_config
        )
        print(analyze_popqa_base_support(config, settings, force=args.force))
    elif args.command == "collect-training-shift":
        settings = load_training_shift_config(args.training_shift_config)
        print(collect_training_shift(config, settings, args.source, force=args.force))
    elif args.command == "analyze-training-shift":
        settings = load_training_shift_config(args.training_shift_config)
        print(analyze_training_shift(config, settings, force=args.force))


if __name__ == "__main__":
    main()
