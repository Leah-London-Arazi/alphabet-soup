import argparse

from utils.utils import set_random_seed, random_sentence, disable_warnings, init_logger

disable_warnings()
set_random_seed()

from tqdm import trange
from textattack.metrics import Perplexity, AttackQueries, AttackSuccessRate
from omegaconf import OmegaConf
from metrics.entropy import Entropy
from metrics.time import Time
from utils.attack import run_attack
from utils.recipes import get_attack_recipe_from_args
from utils.utils import get_logger


logger = get_logger(__name__)


def log_metrics(results, metrics, extra=None):
    metrics_results = []
    for metric in metrics:
        try:
            metric_result = metric().calculate(results)
            if metric_result:
                metrics_results.append(metric_result)
        except Exception as e:
            logger.error(f"Caught exception while calculating metrics: {e}", extra=extra)
            continue

    logger.info(f"Metric results where written to file: {metrics_results}", extra=extra)


def run_single_experiment(experiment_num, experiment_args, metrics):
    extra = dict(experiment_num=experiment_num, experiment_args=experiment_args)
    logger.info(f"Running experiment", extra=extra)

    attack_recipe = get_attack_recipe_from_args(experiment_args, from_command_line=False)

    expr_results = []

    for i in trange(experiment_args.num_repetitions):
        attack = attack_recipe.get_attack()
        print(i)

        if experiment_args.rand_init_text:
            init_text = random_sentence()
        else:
            init_text = experiment_args.initial_text
        try:
            expr_rep_result = run_attack(attack=attack, input_text=init_text)
        except Exception as e:
            logger.error(f"Caught exception while running experiment: {e}")
            continue

        expr_results.append(expr_rep_result)

    log_metrics(results=expr_results, metrics=metrics, extra=extra)


def run_experiments(metrics, config_file):
    config = OmegaConf.load(config_file)

    for experiment_num, experiment_config in enumerate(config.experiments):
        experiment_args = OmegaConf.merge(config.defaults, experiment_config)
        for model_name in experiment_args.model_names:
            for target_class in experiment_args.target_classes:
                experiment_args.model_name = model_name
                experiment_args.target_class = target_class
                run_single_experiment(experiment_num=experiment_num,
                                      experiment_args=experiment_args,
                                      metrics=metrics)


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

    run_experiments(metrics=[Entropy, Perplexity, AttackQueries, AttackSuccessRate, Time],
                    config_file=args.config_file)


if __name__ == '__main__':
    main()
