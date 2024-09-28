from textattack.metrics import Perplexity, AttackQueries, AttackSuccessRate
from utils.utils import set_random_seed, random_sentence
from consts import ATTACK_NAME_TO_RECIPE, ATTACK_NAME_TO_PARAMS, AttackName
from metrics.entropy import Entropy
from utils.attack import run_attack
from omegaconf import OmegaConf

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
                                      debug=args.debug,
                                      attack_params=attack_params)
    return attack_recipe

def run_single_experiment(args, metrics):
    attack_results = []
    metrics_results = []
    expr_times = []
    for _ in range(args.num_repetitions):
        attack_recipe = get_attack_recipe(args)
        attack = attack_recipe.get_attack()

        if args.rand_init_text:
            init_text = random_sentence()
        else:
            init_text = args.initial_text

        expr_result, expr_time = run_attack(attack=attack, input_text=init_text)
        attack_results.append(expr_result)
        expr_times.append(expr_time)

    for metric in metrics:
        metrics_results.append(metric().calculate(attack_results))

    metrics_results.append({"avg_attack_time_secs": round(sum(expr_times) / len(expr_times), 2)})

    return attack_results, metrics_results


def run_experiments(metrics):
    set_random_seed()
    config = OmegaConf.load("config.yaml")
    for experiment_config in config.experiments:
        experiment_args = OmegaConf.merge(config.defaults, experiment_config)
        _, metric_results = run_single_experiment(experiment_args, metrics)
        print(f"Metrics for {experiment_args.attack_name}: {metric_results}")

if __name__ == '__main__':
    run_experiments([Entropy, Perplexity, AttackQueries, AttackSuccessRate])
