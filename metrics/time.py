from textattack.attack_results import FailedAttackResult, SkippedAttackResult
from textattack.metrics import Metric


class Time(Metric):
    def __init__(self):
        self.all_metrics = {}

    def calculate(self, results):
        time_results = []
        for result in results:
            if isinstance(result, (FailedAttackResult, SkippedAttackResult)):
                continue
            else:
                if hasattr(result, "attack_time"):
                    time_results.append(result.attack_time)

        if len(time_results) != 0:
            self.all_metrics["avg_attack_time_secs"] = round(
                sum(time_results) / len(time_results), 2)

        return self.all_metrics
