from textattack.metrics import Perplexity, AttackQueries, AttackSuccessRate

from consts import AttackName, MetricName
from metrics.entropy import Entropy
from metrics.score import Score
from metrics.time import Time
from recipes.baseline import Baseline
from schemas import (CharacterRouletteBlackBoxAttackParams,
                     CharacterRouletteWhiteBoxAttackParams,
                     PEZAttackParams,
                     GCGAttackParams,
                     AttackParams)
from recipes.character_roulette import CharacterRouletteBlackBoxRandomChar, CharacterRouletteBlackBoxRandomWord, CharacterRouletteWhiteBox
from recipes.pez import PEZ
from recipes.gcg import GCG


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


METRIC_NAME_TO_CLASS = {
    MetricName.entropy: Entropy,
    MetricName.perplexity: Perplexity,
    MetricName.queries: AttackQueries,
    MetricName.success_rate: AttackSuccessRate,
    MetricName.time: Time,
    MetricName.score: Score
}


def get_attack_recipe_from_args(args, from_command_line=False):
    attack_name = AttackName(args.attack_name)
    attack_recipe_cls = ATTACK_NAME_TO_RECIPE[attack_name]
    attack_params_cls = ATTACK_NAME_TO_PARAMS[attack_name]

    attack_params = args.get('attack_params', {})
    attack_params = attack_params_cls._from_command_line_args(attack_params) if from_command_line else attack_params_cls(**attack_params)

    attack_recipe = attack_recipe_cls(model_name=args.model_name,
                                      targeted=args.targeted,
                                      target_class=args.target_class if args.targeted else 0,
                                      confidence_threshold=args.confidence_threshold,
                                      query_budget=args.query_budget,
                                      attack_params=attack_params)
    return attack_recipe



