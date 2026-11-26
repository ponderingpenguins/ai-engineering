"""Load the baked radar dataset, cut labelled windows and run person-wise experiments.

cv holds out people for cross-validation. tune selects hyperparameters inside the
training folds. train-final fits one model on all available people using the saved
tuned setting. bake builds the dataset from raw recordings and source annotations.
Settings come from config/data.yaml, config/train.yaml and config/models/<name>.yaml;
command-line key=value overrides take precedence.

Examples:
    python main.py cv model=baseline cv.folds=2
    python main.py tune 'tune.grid=[{model.arch.hidden: 64}, {model.arch.hidden: 128, training.lr: 0.0003}]'
    python main.py train-final
    python main.py bake --num-proc 4 --overwrite
    python main.py bake --refresh-annotations
"""
import argparse
import logging
import sys
from pathlib import Path
from omegaconf.errors import OmegaConfBaseException
from config import load_run_config
from training.pipelines import run_cv, run_train_final, run_tune

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parent


def main() -> None:
    """Parse the command and config, then run the requested experiment or bake."""
    if len(sys.argv) > 1 and sys.argv[1] == "bake":
        from data.bake import main as bake
        sys.argv = [f"{Path(sys.argv[0]).name} bake", *sys.argv[2:]]
        bake()
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("cv", "tune", "train-final", "bake"))
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
