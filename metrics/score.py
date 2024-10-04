from textattack.attack_results import FailedAttackResult, SkippedAttackResult, SuccessfulAttackResult
from textattack.metrics import Metric
import numpy as np

class Score(Metric):
    def __init__(self, include_skipped_results=False):
        self.all_metrics = {}
        self.include_skipped_results = include_skipped_results


    def calculate(self, results):
        failed_attack_score = []
        succeeded_attack_score = []
        failed_attack_classified_as = []
        failed_attack_scores_per_class = {}

        for result in results:
            if isinstance(result, SuccessfulAttackResult):
                succeeded_attack_score.append(result.perturbed_result.score)
            if isinstance(result, FailedAttackResult):
                failed_attack_score.append(result.perturbed_result.score)
                failed_attack_classified_as.append(result.perturbed_result.output)

                model_output = result.perturbed_result.output
                if not failed_attack_scores_per_class.get(model_output):
                    failed_attack_scores_per_class[model_output] = []
                failed_attack_scores_per_class[model_output].append(result.perturbed_result.score)

                failed_attack_classified_as.append(model_output)

            if isinstance(result, SkippedAttackResult) and not self.include_skipped_results:
                continue

        avg_failed_score_per_class = {}
        for output_class, scores in failed_attack_scores_per_class.items():
            if len(scores) > 0:
                avg_failed_score_per_class[output_class] = round(
                    (sum(scores) / len(scores)), 2)

        self.all_metrics["avg_failed_score_per_class"] = avg_failed_score_per_class

        if len(succeeded_attack_score) > 0:
            self.all_metrics["avg_succeeded_attack_score"] = round(
                sum(succeeded_attack_score) / len(succeeded_attack_score), 2)

        if len(failed_attack_score) > 0:
            self.all_metrics["avg_failed_attack_score"] = round(
                sum(failed_attack_score) / len(failed_attack_score), 2)

        if len(failed_attack_classified_as) > 0:
            self.all_metrics["failed_attack_prediction_dist"] = (
                    np.bincount(failed_attack_classified_as) / len(failed_attack_classified_as))

        return self.all_metrics
