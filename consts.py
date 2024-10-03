import enum

class FilterTokenIDsMethod(enum.Enum):
    by_target_class = "by_target_class"
    by_bert_score = "by_bert_score"
    by_glove_score = "by_glove_score"
    by_random_tokens = "by_random_tokens"


class AttackName(enum.Enum):
    baseline = "baseline"
    character_roulette_black_box_random_char = "character_roulette_black_box_random_char"
    character_roulette_black_box_random_word = "character_roulette_black_box_random_word"
    character_roulette_white_box = "character_roulette_white_box"
    pez = "pez"
    gcg = "gcg"


class MetricName(enum.Enum):
    ENTROPY = "entropy"
    PERPLEXITY = "perplexity"
    QUERIES = "queries"
    SUCCESS_RATE = "success_rate"
    TIME = "time"
    SCORE = "score"
