import enum

class FilterTokenIDsMethod(enum.Enum):
    by_target_class = "by_target_class"
    by_bert_score = "by_bert_score"
    by_glove_score = "by_glove_score"


class AttackName(enum.Enum):
    character_roulette_black_box_random_char = "character_roulette_black_box_random_char"
    character_roulette_black_box_random_word = "character_roulette_black_box_random_word"
    character_roulette_white_box = "character_roulette_white_box"
    pez= "pez"
    gcg = "gcg"


from schemas import CharacterRouletteBlackBoxAttackParams, CharacterRouletteWhiteBoxAttackParams, PEZAttackParams, \
    GCGAttackParams
from recipes import CharacterRouletteBlackBoxRandomChar, CharacterRouletteBlackBoxRandomWord, \
    CharacterRouletteWhiteBox, PEZ, GCG


ATTACK_NAME_TO_RECIPE = {
    AttackName.character_roulette_black_box_random_char: CharacterRouletteBlackBoxRandomChar,
    AttackName.character_roulette_black_box_random_word: CharacterRouletteBlackBoxRandomWord,
    AttackName.character_roulette_white_box: CharacterRouletteWhiteBox,
    AttackName.pez: PEZ,
    AttackName.gcg: GCG,
}


ATTACK_NAME_TO_PARAMS = {
    AttackName.character_roulette_black_box_random_char: CharacterRouletteBlackBoxAttackParams,
    AttackName.character_roulette_black_box_random_word: CharacterRouletteBlackBoxAttackParams,
    AttackName.character_roulette_white_box: CharacterRouletteWhiteBoxAttackParams,
    AttackName.pez: PEZAttackParams,
    AttackName.gcg: GCGAttackParams,
}