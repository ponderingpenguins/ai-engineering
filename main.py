"""Run radar data preparation, training, annotation and reporting commands.

Commands: cv, tune, train-final, bake, report, annotate, audit.
Defaults: config/data.yaml, config/train.yaml and config/models/<name>.yaml.
Command-line key=value overrides take precedence, see README.md for examples.
"""

import argparse
import logging
import sys
from importlib import import_module
from pathlib import Path
from omegaconf.errors import OmegaConfBaseException
from config.loader import load_run_config

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parent
TOOL_COMMANDS = {
    "bake": "data.bake",
    "report": "evaluation.report",
    "annotate": "annotation_tool.__main__",
    "audit": "evaluation.annotation_audit",
}


def main() -> None:
    """Dispatch tools to their own parsers, or load the config for a training pipeline."""
    if len(sys.argv) > 1 and sys.argv[1] in TOOL_COMMANDS:
        command = sys.argv[1]
        tool_main = import_module(TOOL_COMMANDS[command]).main
        original_argv = sys.argv
        sys.argv = [f"{Path(original_argv[0]).name} {command}", *original_argv[2:]]
        try:
            tool_main()
        finally:
            sys.argv = original_argv
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("cv", "tune", "train-final", *TOOL_COMMANDS))
    args, overrides = parser.parse_known_args()
    try:
        cfg = load_run_config(REPO, overrides)
        if args.command == "tune" and not cfg.tune.grid:
            raise ValueError("tune requires a non-empty tune.grid")
    except (TypeError, ValueError, OmegaConfBaseException) as error:
        parser.error(str(error))
    from training.pipelines import run_cv, run_train_final, run_tune

    try:
        {"cv": run_cv, "tune": run_tune, "train-final": run_train_final}[args.command](cfg, REPO)
    except ValueError as error:
        logger.error("%s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
