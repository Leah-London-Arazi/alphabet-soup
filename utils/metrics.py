from consts import MetricName
from textattack.metrics import Perplexity, AttackQueries, AttackSuccessRate
from metrics.entropy import Entropy
from metrics.score import Score
from metrics.time import Time


METRIC_NAME_TO_CLASS = {
    MetricName.ENTROPY: Entropy,
    MetricName.PERPLEXITY: Perplexity,
    MetricName.QUERIES: AttackQueries,
    MetricName.SUCCESS_RATE: AttackSuccessRate,
    MetricName.TIME: Time,
    MetricName.SCORE: Score
}
