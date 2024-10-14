from textattack.attack_results import FailedAttackResult, SkippedAttackResult, SuccessfulAttackResult
from textattack.metrics import Metric
import numpy as np

class Score(Metric):
    def __init__(self):
        self.all_metrics = {}


    def calculate(self, results):
        failed_attack_score = []
        succeeded_attack_score = []
        failed_attack_classified_as = []
        skipped_attack_score = []
        skipped_attack_classified_as = []

        for result in results:
            if isinstance(result, SuccessfulAttackResult):
                succeeded_attack_score.append(result.perturbed_result.score)

            if isinstance(result, FailedAttackResult):
                failed_attack_score.append(result.perturbed_result.score)
                failed_attack_classified_as.append(result.perturbed_result.output)

            if isinstance(result, SkippedAttackResult):
                skipped_attack_score.append(result.perturbed_result.score)
                skipped_attack_classified_as.append(result.perturbed_result.output)

        if len(succeeded_attack_score) > 0:
            self.all_metrics["avg_succeeded_attack_score"] = round(
                sum(succeeded_attack_score) / len(succeeded_attack_score), 2)

        if len(failed_attack_score) > 0:
            self.all_metrics["avg_failed_attack_score"] = round(
                sum(failed_attack_score) / len(failed_attack_score), 2)

        if len(failed_attack_classified_as) > 0:
            self.all_metrics["failed_attack_prediction_dist"] = (
                    np.bincount(failed_attack_classified_as) / len(failed_attack_classified_as))
        
        if len(skipped_attack_score) > 0:
            self.all_metrics["skipped_attack_score"] = round(
                sum(skipped_attack_score) / len(skipped_attack_score), 2)

        if len(skipped_attack_classified_as) > 0:
            self.all_metrics["skipped_attack_classified_as"] = (
                    np.bincount(skipped_attack_classified_as) / len(skipped_attack_classified_as))


        return self.all_metrics
