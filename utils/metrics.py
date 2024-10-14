from consts import MetricName
from textattack.metrics import AttackSuccessRate
from metrics.perplexity import Perplexity
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


def get_metrics_from_config(metrics_config):
    metrics = []
    for metric in metrics_config:
        metric_class = METRIC_NAME_TO_CLASS[MetricName(metric.name)]
        metrics_args = metric.get("metric_params", {})
        metrics.append(metric_class(**metrics_args))
    return metrics
