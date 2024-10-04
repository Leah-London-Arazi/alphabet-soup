from consts import MetricName
from textattack.metrics import Perplexity, AttackSuccessRate
from metrics.entropy import Entropy
from metrics.score import Score
from metrics.time import Time
from metrics.queries import Queries


METRIC_NAME_TO_CLASS = {
    MetricName.ENTROPY: Entropy,
    MetricName.PERPLEXITY: Perplexity,
    MetricName.QUERIES: Queries,
    MetricName.SUCCESS_RATE: AttackSuccessRate,
    MetricName.TIME: Time,
    MetricName.SCORE: Score
}
