"""Guard CLI import and execution with persistent diagnostics on an existing run."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peft_probe.factual_runtime import run_job


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="configs/factual_shift.yaml")
    args, _ = parser.parse_known_args()

    def invoke():
        from peft_probe.cli import main as cli_main

        return cli_main()

    run_job(args.config, invoke)


if __name__ == "__main__":
    main()
