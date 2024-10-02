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
    entropy = "entropy"
    perplexity = "perplexity"
    queries = "queries"
    success_rate = "success_rate"
    time = "time"
    score = "score"
