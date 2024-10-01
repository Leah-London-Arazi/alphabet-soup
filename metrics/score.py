from textattack.attack_results import FailedAttackResult, SkippedAttackResult
from textattack.metrics import Metric
import numpy as np

class Score(Metric):
    def __init__(self, include_skipped_results=True):
        self.all_metrics = {}
        self.include_skipped_results = include_skipped_results


    def calculate(self, results):
        scores = []
        classified_as = []
        for result in results:
            if isinstance(result, FailedAttackResult):
                continue
            if isinstance(result, SkippedAttackResult) and not self.include_skipped_results:
                continue
            scores.append(result.perturbed_result.score)
            classified_as.append(result.perturbed_result.output)

        if len(scores) != 0:
            self.all_metrics["avg_attack_score"] = round(
                sum(scores) / len(scores), 2)

        if len(classified_as) != 0:
            self.all_metrics["class_dist"] = np.bincount(classified_as) / len(classified_as)

        return self.all_metrics
