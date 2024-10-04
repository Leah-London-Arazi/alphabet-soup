from consts import AttackName
from recipes.baseline import Baseline
from schemas import (CharacterRouletteBlackBoxAttackParams,
                     CharacterRouletteWhiteBoxAttackParams,
                     PEZAttackParams,
                     GCGAttackParams,
                     AttackParams)
from recipes.character_roulette import CharacterRouletteBlackBoxRandomChar, CharacterRouletteBlackBoxRandomWord, CharacterRouletteWhiteBox
from recipes.pez import PEZ
from recipes.gcg import GCG
from timeit import default_timer as timer
from utils.utils import get_logger

logger = get_logger(__name__)

ATTACK_NAME_TO_RECIPE = {
    AttackName.character_roulette_black_box_random_char: CharacterRouletteBlackBoxRandomChar,
    AttackName.character_roulette_black_box_random_word: CharacterRouletteBlackBoxRandomWord,
    AttackName.character_roulette_white_box: CharacterRouletteWhiteBox,
    AttackName.pez: PEZ,
    AttackName.gcg: GCG,
    AttackName.baseline: Baseline
}


ATTACK_NAME_TO_PARAMS = {
    AttackName.character_roulette_black_box_random_char: CharacterRouletteBlackBoxAttackParams,
    AttackName.character_roulette_black_box_random_word: CharacterRouletteBlackBoxAttackParams,
    AttackName.character_roulette_white_box: CharacterRouletteWhiteBoxAttackParams,
    AttackName.pez: PEZAttackParams,
    AttackName.gcg: GCGAttackParams,
    AttackName.baseline: AttackParams
}


def get_attack_recipe_from_args(args, from_command_line=False):
    attack_name = AttackName(args.attack_name)
    attack_recipe_cls = ATTACK_NAME_TO_RECIPE[attack_name]
    attack_params_cls = ATTACK_NAME_TO_PARAMS[attack_name]

    if from_command_line:
        attack_params = attack_params_cls._from_command_line_args(args.attack_params)
    else:
        attack_params = attack_params_cls(**args.get('attack_params', {}))

    attack_recipe = attack_recipe_cls(model_name=args.model_name,
                                      targeted=args.targeted,
                                      target_class=args.target_class,
                                      confidence_threshold=args.confidence_threshold,
                                      query_budget=args.query_budget,
                                      attack_params=attack_params)
    return attack_recipe


def run_attack(attack, input_text, label=1):
    start = timer()
    attack_result = attack.attack(input_text, label)
    attack_result.attack_time = timer() - start
    logger.log_final_result(attack_result)
    return attack_result
