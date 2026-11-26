"""Load the baked radar dataset, cut labelled windows and run person-wise experiments.

cv holds out people for cross-validation. tune selects hyperparameters inside the
training folds. train-final fits one model on all available people using the saved
tuned setting. bake builds the dataset from raw recordings and source annotations.
report rebuilds a saved run's report. annotate opens the annotation tool, and audit
compares reviewed annotations with the original pose labels.
Settings come from config/data.yaml, config/train.yaml and config/models/<name>.yaml;
command-line key=value overrides take precedence.

Examples:
    python main.py cv model=baseline cv.folds=2
    python main.py tune 'tune.grid=[{model.arch.hidden: 64}, {model.arch.hidden: 128, training.lr: 0.0003}]'
    python main.py train-final
    python main.py bake --num-proc 4 --overwrite
    python main.py bake --refresh-annotations
    python main.py report results/<run>
    python main.py annotate --no-browser
    python main.py audit --out results/annotation_agreement.md
"""
import argparse
import logging
import sys
from importlib import import_module
from pathlib import Path
from omegaconf.errors import OmegaConfBaseException
from config import load_run_config
from training.pipelines import run_cv, run_train_final, run_tune

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
    try:
        {"cv": run_cv, "tune": run_tune, "train-final": run_train_final}[args.command](cfg, REPO)
    except ValueError as error:
        logger.error("%s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
