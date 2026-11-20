"""
Loads the baked dataset, cuts it into windows and cross-validates a model, person by person.

Everything is configured in config/config.yaml and can be overridden from the command line.
By default there is one fold per person; set eval.folds=<k> for k-fold instead:

    python main.py model.name=baseline
    python main.py model.rnn.epochs=5 eval.folds=3 eval.collapse_lying=false

Tune first, then train the final model with the best setting (see "Evaluation workflow" in the README):

    python main.py 'tune.grid=[{hidden: 64}, {hidden: 128, lr: 0.0003}]'
    python main.py train.full_run=true
"""

import logging
import sys
from pathlib import Path

from omegaconf.errors import OmegaConfBaseException

from models.factory import MODELS
from utils.config import load_config
from utils.experiment import check_config, run

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent


def main() -> None:
    """Reads the configuration, runs the experiment and turns errors into exit codes."""
    try:
        cfg = load_config(REPO, sys.argv[1:])  # unknown keys and bad types raise here
        check_config(cfg)
    except (TypeError, ValueError, OmegaConfBaseException) as e:  # pylint: disable=broad-exception-raised
        logger.error(
            "Error: %s\n\nUsage: python main.py [data.baked=<path>] [model.name=%s] "
            "[model.rnn.epochs=<n>] [train.oversample=<ratio>] [train.validation=true|false] [model.rnn.augment.range_shift_bins=<n>] [model.pointnet.augment.shift_m=<m>] [eval.folds=<k>] [eval.collapse_lying=true|false] [tune.grid=<list of settings>] [train.full_run=true|false]",
            e,
            "|".join(MODELS),
        )
        sys.exit(1)
    try:
        run(cfg, REPO)
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
