from omegaconf.errors import ConfigKeyError

from utils.utils import set_random_seed, random_sentence, disable_warnings, init_logger, create_dir, get_current_time, \
    get_escaped_model_name

disable_warnings()
set_random_seed()

import numpy as np
from textattack.datasets import HuggingFaceDataset
import gc
import argparse
import os.path
import traceback
from tqdm import trange
from omegaconf import OmegaConf
from experiment.experiment import Experiment
from utils.recipes import get_attack_recipe_from_args, run_attack
from utils.utils import get_logger

METRICS_RESULTS_DIR_NAME = "metrics_results"


logger = get_logger(__name__)


def create_results_dir():
    current_time = get_current_time()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(current_dir, METRICS_RESULTS_DIR_NAME, current_time)
    create_dir(dir_path)
    return dir_path


def run_experiments(config_file):
    config = OmegaConf.load(config_file)

    base_results_dir = create_results_dir()

    for experiment_config in config.experiments:
        experiment_args = OmegaConf.merge(config.defaults, experiment_config)

        for model in experiment_args.models:
            try:
                target_classes = OmegaConf.merge(model, experiment_args).target_classes
            except ConfigKeyError:
                target_classes = [0]

            for target_class in target_classes:
                experiment_name = experiment_args.name
                attack_recipe_args = experiment_args.attack_recipe
                attack_recipe_args.model_name = model.name
                attack_recipe_args.target_class = target_class
                input_text = experiment_args.get("input_text")
                dataset = experiment_args.get("dataset")
                num_repetitions = experiment_args.num_repetitions

                attack_recipe = get_attack_recipe_from_args(attack_recipe_args, from_command_line=False)
                
                experiment = Experiment(name=experiment_name,
                                        attack_recipe=attack_recipe,
                                        input_text=input_text,
                                        dataset=dataset,
                                        num_repetitions=num_repetitions,
                                        metrics_config = config.metrics,
                                        base_metrics_results_dir=base_results_dir, )


                experiment.run()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True,
                        help="experiments configuration file")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="logging level")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    init_logger(level_name=args.log_level)

    run_experiments(config_file=args.config_file)


if __name__ == '__main__':
    main()
