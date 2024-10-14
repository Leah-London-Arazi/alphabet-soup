from textattack.attack_results import FailedAttackResult, SkippedAttackResult
from textattack.metrics.quality_metrics import Perplexity as TextAttackPerplexity


class Perplexity(TextAttackPerplexity):
    def calculate(self, results):
        self.results = results
        self.original_candidates_ppl = []
        self.successful_candidates_ppl = []
        self.skipped_candidates_ppl = []

        for i, result in enumerate(self.results):
            if isinstance(result, FailedAttackResult):
                continue
            elif isinstance(result, SkippedAttackResult):
                self.skipped_candidates_ppl.append(result.perturbed_result.attacked_text.text.lower())
            else:
                self.original_candidates.append(
                    result.original_result.attacked_text.text.lower()
                )
                self.successful_candidates.append(
                    result.perturbed_result.attacked_text.text.lower()
                )

        ppl_orig = self.calc_ppl(self.original_candidates)
        ppl_attack = self.calc_ppl(self.successful_candidates)
        ppl_skipped = self.calc_ppl(self.skipped_candidates_ppl)

        if len(ppl_orig) > 0:
            self.all_metrics["avg_original_perplexity"] = round(ppl_orig, 2)

        if len(ppl_attack) > 0:
            self.all_metrics["avg_attack_perplexity"] = round(ppl_attack, 2)

        if len(ppl_skipped) > 0:
            self.all_metrics["avg_skipped_perplexity"] = round(ppl_skipped, 2)

        return self.all_metrics
