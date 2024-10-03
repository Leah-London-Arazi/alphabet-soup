from utils.utils import set_random_seed, random_sentence, disable_warnings, init_logger, create_dir, get_current_time, \
    get_escaped_model_name

disable_warnings()
set_random_seed()

import argparse
import os.path
import traceback
from tqdm import trange
from omegaconf import OmegaConf
from consts import MetricName
from utils.attack import run_attack
from utils.recipes import get_attack_recipe_from_args, METRIC_NAME_TO_CLASS
from utils.utils import get_logger

METRICS_RESULTS_DIR_NAME = "metrics_results"


logger = get_logger(__name__)


def create_metrics_dir():
    current_time = get_current_time()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(current_dir, METRICS_RESULTS_DIR_NAME, current_time)
    create_dir(dir_path)
    return dir_path


def _write_metrics_to_file(experiment_num, experiment_args, experiment_info, metrics_dir, metrics_results):
    results_file_name = (f"experiment_num={experiment_num}"
                         f"_model_name={get_escaped_model_name(experiment_args.model_name)}"
                         f"_target_class={experiment_args.target_class}")
    results_file_path = os.path.join(metrics_dir, results_file_name)
    with open(os.path.join(results_file_path), "w") as f:
        f.write(f"experiment_args={experiment_args}\nmetrics_results={metrics_results}")

    logger.info(f"Metric results were written to file: {results_file_path}", extra=experiment_info)


def calculate_metrics(results, metrics, metrics_dir, experiment_num, experiment_args):
    experiment_info = dict(experiment_num=experiment_num, experiment_args=experiment_args)

    metrics_results = []
    for metric in metrics:
        try:
            metric_result = metric().calculate(results)
            if metric_result:
                metrics_results.append(metric_result)
        except:
            logger.error(f"Caught exception while calculating metric {metric.__name__}: "
                         f"{traceback.format_exc()}", extra=experiment_info)
            continue

    _write_metrics_to_file(experiment_num, experiment_args, experiment_info, metrics_dir, metrics_results)


def run_single_experiment(experiment_num, experiment_args, metrics, metrics_dir):
    experiment_info = dict(experiment_num=experiment_num, experiment_args=experiment_args)
    logger.update_extra(extra=experiment_info)
    logger.info(f"Running experiment", extra=experiment_info)

    attack_recipe = get_attack_recipe_from_args(experiment_args, from_command_line=False)

    expr_results = []

    for _ in trange(experiment_args.num_repetitions):
        attack = attack_recipe.get_attack()

        if experiment_args.rand_init_text:
            init_text = random_sentence()
        else:
            init_text = experiment_args.initial_text
        try:
            expr_rep_result = run_attack(attack=attack, input_text=init_text)
        except:
            logger.error(f"Caught exception while running experiment: {traceback.format_exc()}")
            continue

        expr_results.append(expr_rep_result)

    calculate_metrics(results=expr_results,
                      metrics=metrics,
                      metrics_dir=metrics_dir,
                      experiment_num=experiment_num,
                      experiment_args=experiment_args)


def run_experiments(config_file):
    config = OmegaConf.load(config_file)

    metrics = [METRIC_NAME_TO_CLASS[MetricName(metric)] for metric in config.metrics]
    metrics_dir = create_metrics_dir()

    for experiment_num, experiment_config in enumerate(config.experiments):
        experiment_args = OmegaConf.merge(config.defaults, experiment_config)
        # create metrics directory for each attack
        attack_metrics_dir = os.path.join(metrics_dir, experiment_args.attack_name)
        create_dir(attack_metrics_dir)

        for model in experiment_args.models:
            if not experiment_args.targeted:
                experiment_args.target_classes = [0]
            for target_class in OmegaConf.merge(model, experiment_args).target_classes:
                experiment_args.model_name = model.name
                experiment_args.target_class = target_class
                run_single_experiment(experiment_num=experiment_num,
                                      experiment_args=experiment_args,
                                      metrics=metrics,
                                      metrics_dir=attack_metrics_dir)


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
