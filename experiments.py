import argparse

from utils.utils import set_random_seed, random_sentence, disable_warnings, init_logger, get_root_logger

disable_warnings()

from tqdm import trange
from textattack.metrics import Perplexity, AttackQueries, AttackSuccessRate
from omegaconf import OmegaConf
from consts import ATTACK_NAME_TO_RECIPE, ATTACK_NAME_TO_PARAMS, AttackName
from metrics.entropy import Entropy
from metrics.time import Time
from metrics.score import Score
from utils.attack import run_attack


def get_attack_recipe(args):
    attack_name = AttackName(args.attack_name)
    attack_recipe_cls = ATTACK_NAME_TO_RECIPE[attack_name]
    attack_params_cls = ATTACK_NAME_TO_PARAMS[attack_name]
    if "attack_params" not in args:
        args.attack_params = {}
    attack_params = attack_params_cls(**args.attack_params)

    attack_recipe = attack_recipe_cls(model_name=args.model_name,
                                      targeted=args.targeted,
                                      target_class=args.target_class,
                                      confidence_threshold=args.confidence_threshold,
                                      query_budget=args.query_budget,
                                      attack_params=attack_params)
    return attack_recipe


def log_metrics(results, metrics, args, experiment_number, logger):
    metrics_results = []
    for metric in metrics:
        try:
            metric_result = metric().calculate(results)
            if metric_result:
                metrics_results.append(metric_result)
        except Exception as e:
            logger.error(f"Caught exception while calculating "
                         f"metrics in {args.attack_name} on {args.model_name}: {e}")
            continue

    logger.info(f"Metric results for experiment number {experiment_number}: "
                f"attack {args.attack_name} on {args.model_name}: {metrics_results}")


def run_single_experiment(args, metrics, experiment_number):
    logger = get_root_logger()
    logger.info(f"Experiment {experiment_number} configuration: {args}")
    expr_results = []
    logger.info(f"Running experiment number {experiment_number}: "
                f"attack {args.attack_name} on {args.model_name} "
                f"for {args.num_repetitions} repetitions")
    for rep_i in trange(args.num_repetitions):
        logger.debug(f"Experiment {experiment_number} repetition {rep_i}")
        attack_recipe = get_attack_recipe(args)
        attack = attack_recipe.get_attack()

        if args.rand_init_text:
            init_text = random_sentence()
        else:
            init_text = args.initial_text
        try:
            expr_rep_result = run_attack(attack=attack, input_text=init_text)
        except Exception as e:
            logger.error(f"Caught exception while running "
                         f"attack {args.attack_name} on {args.model_name}: {e}")
            continue
        expr_results.append(expr_rep_result)

    log_metrics(expr_results, metrics, args, experiment_number, logger)

def run_experiments(metrics, config_file):
    set_random_seed()
    config = OmegaConf.load(config_file)
    init_logger(level_name=config.defaults.log_level)
    for experiment_num, experiment_config in enumerate(config.experiments):
        experiment_args = OmegaConf.merge(config.defaults, experiment_config)
        for model_name in experiment_args.model_names:
            for target_class in experiment_args.target_classes:
                experiment_args.model_name = model_name
                experiment_args.target_class = target_class
                run_single_experiment(experiment_args, metrics, experiment_num)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True,
                        help="experiments configuration file")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    run_experiments([Entropy, Perplexity, AttackQueries, AttackSuccessRate, Time, Score], args.config_file)


if __name__ == '__main__':
    main()
