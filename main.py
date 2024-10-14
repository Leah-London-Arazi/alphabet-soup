from utils.utils import disable_warnings

disable_warnings()

import argparse
from consts import AttackName
from utils.recipes import get_attack_recipe_from_args, run_attack
from utils.utils import random_sentence, init_logger
from utils.utils import get_logger

logger = get_logger(__name__)


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
    parser.add_argument("--target-class", type=int, default=0,
                        help="Attacked class")
    parser.add_argument("--query-budget", type=int, default=500,
                        help="Maximal queries allowed to the model")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="logging level")
    parser.add_argument("--attack-params", nargs="+", default=[],
                        help="Additional key=value parameters. "
                             "For lists, enter the values as a comma-separated strings.")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    init_logger(level_name=args.log_level)

    attack_recipe = get_attack_recipe_from_args(args, from_command_line=True)
    attack = attack_recipe.get_attack()
    logger.info("Running attack...")
    run_attack(attack=attack, input_text=args.input_text)


if __name__ == '__main__':
    main()
