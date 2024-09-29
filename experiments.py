import argparse

from utils.utils import set_random_seed, random_sentence, disable_warnings, init_logger, get_root_logger

disable_warnings()

from tqdm import trange
from textattack.metrics import Perplexity, AttackQueries, AttackSuccessRate
from omegaconf import OmegaConf
from consts import ATTACK_NAME_TO_RECIPE, ATTACK_NAME_TO_PARAMS, AttackName
from metrics.entropy import Entropy
from utils.attack import run_attack


def get_attack_recipe(args):
    attack_name = AttackName(args.attack_name)
    attack_recipe_cls = ATTACK_NAME_TO_RECIPE[attack_name]
    attack_params_cls = ATTACK_NAME_TO_PARAMS[attack_name]
    attack_params = attack_params_cls(**args.attack_params)

    attack_recipe = attack_recipe_cls(model_name=args.model_name,
                                      targeted=args.targeted,
                                      target_class=args.target_class,
                                      confidence_threshold=args.confidence_threshold,
                                      query_budget=args.query_budget,
                                      attack_params=attack_params)
    return attack_recipe

def run_single_experiment(args, metrics, experiment_number):
    logger = get_root_logger()
    attack_results = []
    experiment_metrics = {}
    for model_name in args.model_names:
        per_model_time = []
        per_model_metrics = []
        args.model_name = model_name
        logger.info(f"Running experiment number {experiment_number}: "
                    f"attack {args.attack_name} on {model_name} "
                    f"for {args.num_repetitions} repetitions")
        for _ in trange(args.num_repetitions):
            attack_recipe = get_attack_recipe(args)
            attack = attack_recipe.get_attack()

            if args.rand_init_text:
                init_text = random_sentence()
            else:
                init_text = args.initial_text

            try:
                expr_result, expr_time = run_attack(attack=attack, input_text=init_text)
            except Exception as e:
                logger.error(f"Caught exception while running "
                             f"attack {args.attack_name} on {model_name}: {e}")
                continue
            attack_results.append(expr_result)
            per_model_time.append(expr_time)

        for metric in metrics:
            try:
                per_model_metrics.append(metric().calculate(attack_results))
            except Exception as e:
                logger.error(f"Caught exception while calculating "
                             f"metrics in {args.attack_name} on {model_name}: {e}")
                continue
        per_model_metrics.append({"avg_attack_time_secs": round(sum(per_model_time) / len(per_model_time), 2)})

        experiment_metrics[model_name] = per_model_metrics
    return experiment_metrics


def run_experiments(metrics, config_file):
    set_random_seed()
    config = OmegaConf.load(config_file)
    init_logger(level_name=config.defaults.log_level)
    for experiment_num, experiment_config in enumerate(config.experiments):
        experiment_args = OmegaConf.merge(config.defaults, experiment_config)
        experiment_results = run_single_experiment(experiment_args, metrics, experiment_num)
        for model_name, experiment_metrics in experiment_results.items():
            print(f"Metrics for {experiment_args.attack_name} on {model_name}: {experiment_metrics}")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True,
                        help="experiments configuration file")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    run_experiments([Entropy, Perplexity, AttackQueries, AttackSuccessRate], args.config_file)


if __name__ == '__main__':
    main()
