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
        failed_attack_score = []
        failed_attack_classified_as = []
        for result in results:
            if isinstance(result, FailedAttackResult):
                failed_attack_score.append(result.perturbed_result.score)
                failed_attack_classified_as.append(result.perturbed_result.output)
            if isinstance(result, SkippedAttackResult) and not self.include_skipped_results:
                continue
            scores.append(result.perturbed_result.score)
            classified_as.append(result.perturbed_result.output)

        if len(scores) > 0:
            self.all_metrics["avg_attack_score"] = round(
                sum(scores) / len(scores), 2)

        if len(classified_as) > 0:
            self.all_metrics["attack_prediction_dist"] = np.bincount(classified_as) / len(classified_as)

        if len(failed_attack_score) > 0:
            self.all_metrics["avg_failed_attack_score"] = round(
                sum(failed_attack_score) / len(failed_attack_score), 2)

        if len(failed_attack_classified_as) > 0:
            self.all_metrics["failed_attack_prediction_dist"] = (
                    np.bincount(failed_attack_classified_as) / len(failed_attack_classified_as))

        return self.all_metrics
