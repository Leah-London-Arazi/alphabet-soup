from textattack.attack_results import FailedAttackResult, SkippedAttackResult, SuccessfulAttackResult
from textattack.metrics import Metric


class Queries(Metric):
    def __init__(self):
        self.all_metrics = {}


    def calculate(self, results):
        succeeded_attack_queries = []
        skipped_attack_queries = []

        for result in results:
            if isinstance(result, SuccessfulAttackResult):
                succeeded_attack_queries.append(result.num_queries)
            if isinstance(result, SkippedAttackResult):
                skipped_attack_queries.append(result.num_queries)
            if isinstance(result, FailedAttackResult):
                continue

        if len(succeeded_attack_queries) > 0:
            self.all_metrics["avg_succeeded_attack_queries"] = round(
                sum(succeeded_attack_queries) / len(succeeded_attack_queries), 2)

        if len(skipped_attack_queries) > 0:
            self.all_metrics["avg_skipped_attack_queries"] = round(
                sum(skipped_attack_queries) / len(skipped_attack_queries), 2)

        return self.all_metrics
