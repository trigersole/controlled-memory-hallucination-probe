"""Analyze completed factual-shift predictions without importing model libraries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peft_probe.within_question import analyze_saved_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Completed factual_shift_v1/results directory")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force", action="store_true", help="Rebuild this diagnostic's reports only"
    )
    args = parser.parse_args()
    print(analyze_saved_predictions(args.results, args.bootstrap_samples, args.seed, args.force))


if __name__ == "__main__":
    main()
