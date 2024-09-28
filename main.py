import argparse
import logging
import sys

from consts import AttackName, ATTACK_NAME_TO_RECIPE, ATTACK_NAME_TO_PARAMS
from utils.attack import run_attack
from utils.defaults import ROOT_LOGGER_NAME
from utils.utils import random_sentence


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-name", type=AttackName, required=True,
                        help="Attack to run")
    parser.add_argument("--confidence-threshold", type=float, default=0.9,
                        help="Minimal classification confidence required for attack success")
    parser.add_argument("--model-name", type=str,
                        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        help="HuggingFace model name")
    parser.add_argument("--input-text", type=str, default=random_sentence(),
                        help="Initial attacked text, the default is a random sentence")
    parser.add_argument("--targeted", type=bool, default=True,
                        help="Is targeted attack")
    parser.add_argument("--target-class", type=int, default=0,
                        help="Attacked class in targeted mode")
    parser.add_argument("--query-budget", type=int, default=500,
                        help="Maximal queries allowed to the model")
    parser.add_argument("--debug", type=bool, default=False,
                        help="Run in debug mode")
    parser.add_argument("--attack-params", nargs="+", default=[],
                        help="Additional key=value parameters. "
                             "For lists, enter the values as a comma-separated strings.")

    return parser


def init_logger(level=logging.INFO):
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level=level)
    logger.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s\n%(message)s\n',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def main():
    parser = get_parser()

    # Parse the arguments
    args = parser.parse_args()

    # Logger
    init_logger(level=logging.DEBUG if args.debug else logging.INFO)

    attack_recipe_cls = ATTACK_NAME_TO_RECIPE[args.attack_name]
    attack_params_cls = ATTACK_NAME_TO_PARAMS[args.attack_name]

    attack_params = attack_params_cls._from_args(args.attack_params)

    attack_recipe = attack_recipe_cls(model_name=args.model_name,
                                      targeted=args.targeted,
                                      target_class=args.target_class,
                                      confidence_threshold=args.confidence_threshold,
                                      query_budget=args.query_budget,
                                      attack_params=attack_params,)

    attack = attack_recipe.get_attack()
    run_attack(attack=attack, input_text=args.input_text)


if __name__ == '__main__':
    main()
