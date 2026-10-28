"""Guard CLI import and execution with persistent diagnostics on an existing run."""

import argparse
import os
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

    directory = None
    if "run-popqa-factual-shift" in sys.argv:
        import yaml

        from peft_probe.config import load_config

        source = load_config(args.config)
        settings_path = Path(os.environ.get("FACTUAL_POPQA_CONFIG", "configs/factual_popqa.yaml"))
        settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
        directory = Path(settings["output_dir"]) / f"seed_{source['experiment']['seed']}"
    run_job(args.config, invoke, directory=directory)


if __name__ == "__main__":
    main()
