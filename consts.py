import enum

from recipes import CharacterRouletteBlackBoxRandomChar, CharacterRouletteBlackBoxRandomWord, \
    CharacterRouletteWhiteBox, UnboundedDriftPEZ, UnboundedDriftGCG
from schemas import CharacterRouletteBlackBoxAttackParams, CharacterRouletteWhiteBoxAttackParams, PEZAttackParams, \
    GCGAttackParams

class FilterTokenIDsMethod(enum.Enum):
    by_target_class = "by_target_class"
    by_bert_score = "by_bert_score"
    by_glove_score = "by_glove_score"

class AttackName(enum.Enum):
    character_roulette_black_box_random_char = "character_roulette_black_box_random_char"
    character_roulette_black_box_random_word = "character_roulette_black_box_random_word"
    character_roulette_white_box = "character_roulette_white_box"
    unbounded_drift_pez= "unbounded_drift_pez"
    unbounded_drift_gcg = "unbounded_drift_gcg"

    # TODO: add more attacks


ATTACK_NAME_TO_RECIPE = {
    AttackName.character_roulette_black_box_random_char: CharacterRouletteBlackBoxRandomChar,
    AttackName.character_roulette_black_box_random_word: CharacterRouletteBlackBoxRandomWord,
    AttackName.character_roulette_white_box: CharacterRouletteWhiteBox,
    AttackName.unbounded_drift_pez: UnboundedDriftPEZ,
    AttackName.unbounded_drift_gcg: UnboundedDriftGCG,
}


ATTACK_NAME_TO_PARAMS = {
    AttackName.character_roulette_black_box_random_char: CharacterRouletteBlackBoxAttackParams,
    AttackName.character_roulette_black_box_random_word: CharacterRouletteBlackBoxAttackParams,
    AttackName.character_roulette_white_box: CharacterRouletteWhiteBoxAttackParams,
    AttackName.unbounded_drift_pez: PEZAttackParams,
    AttackName.unbounded_drift_gcg: GCGAttackParams,
}
