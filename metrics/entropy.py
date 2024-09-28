import torch
from textattack.attack_results import FailedAttackResult, SkippedAttackResult
from textattack.metrics import Metric


class Entropy(Metric):
    def __init__(self):
        self.all_metrics = {}

    def calculate(self, results):
        entropy_results = []
        for result in results:
            if isinstance(result, (FailedAttackResult, SkippedAttackResult)):
                continue
            else:
                entropy_results.append(
                    Entropy.char_level_entropy(result.perturbed_result.attacked_text.text)
                )
        
        if len(entropy_results) != 0:
            self.all_metrics["avg_attack_entropy"] = sum(entropy_results) / len(entropy_results)

        return self.all_metrics


    @staticmethod
    def char_level_entropy(text):
        # Create a set of unique characters
        unique_chars = list(set(text))

        # Convert characters to indices
        indices = torch.tensor(list(range(len(unique_chars))), dtype=torch.long)

        # Count character occurrences
        num_chars = len(text)
        counts = torch.bincount(indices, minlength=len(unique_chars))

        # Convert counts to probabilities
        probabilities = counts.float() / num_chars

        return torch.sum(torch.special.entr(probabilities)).item()
