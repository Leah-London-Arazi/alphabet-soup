import enum
from textattack.metrics import Perplexity, AttackQueries, AttackSuccessRate
from metrics.entropy import Entropy
from metrics.score import Score
from metrics.time import Time


class MetricName(enum.Enum):
    ENTROPY = "entropy"
    PERPLEXITY = "perplexity"
    QUERIES = "queries"
    SUCCESS_RATE = "success_rate"
    TIME = "time"
    SCORE = "score"

METRIC_NAME_TO_CLASS = {
    MetricName.ENTROPY: Entropy,
    MetricName.PERPLEXITY: Perplexity,
    MetricName.QUERIES: AttackQueries,
    MetricName.SUCCESS_RATE: AttackSuccessRate,
    MetricName.TIME: Time,
    MetricName.SCORE: Score
}
