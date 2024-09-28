###########
# Adaptation of beam search method of textAttack.
# https://github.com/QData/TextAttack/blob/master/textattack/search_methods/beam_search.py
###########

import numpy as np

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import BeamSearch as BaseBeamSearch

from utils.utils import get_logger


class BeamSearch(BaseBeamSearch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self.__module__)


    def perform_search(self, initial_result):
        beam = [initial_result.attacked_text]
        self.logger.log_result(result=initial_result)

        best_result = initial_result
        i = 0

        while not best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            potential_next_beam = []
            for text in beam:
                transformations = self.get_transformations(
                    text, original_text=initial_result.attacked_text
                )
                potential_next_beam += transformations

            if len(potential_next_beam) == 0:
                # If we did not find any possible perturbations, give up.
                return best_result
            results, search_over = self.get_goal_results(potential_next_beam)
            scores = np.array([r.score for r in results])
            best_result = results[scores.argmax()]

            self.logger.log_result(i=i, result=best_result)

            i += 1

            if search_over:
                return best_result

            # Refill the beam. This works by sorting the scores
            # in descending order and filling the beam from there.
            best_indices = (-scores).argsort()[: self.beam_width]
            beam = [potential_next_beam[i] for i in best_indices]

        return best_result
