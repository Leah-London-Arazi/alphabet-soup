###########
# Adaptation of Greedy Word Swap with Word Importance Ranking from TextAttack
###########

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import GreedyWordSwapWIR

from utils.utils import get_logger


class GreedyWordSwapThresholdWIR(GreedyWordSwapWIR):
    def __init__(self, wir_method="unk", unk_token="[UNK]", swap_threshold=0.0, num_transformations_per_word=1):
        super().__init__(wir_method=wir_method, unk_token=unk_token)
        self.swap_threshold = swap_threshold
        self.num_transformation_per_word = num_transformations_per_word
        self.logger = get_logger(self.__module__)

    def perform_search(self, initial_result):
        attacked_text = initial_result.attacked_text
        index_order, exhausted_queries = self._get_index_order(attacked_text)
        self.logger.log_result(result=initial_result)

        cur_result = initial_result

        i = 0

        while not exhausted_queries and cur_result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
            for _ in range(self.num_transformation_per_word):
                transformed_text_candidates = self.get_transformations(
                    cur_result.attacked_text,
                    original_text=initial_result.attacked_text,
                    indices_to_modify=[index_order[i]],
                )
                if len(transformed_text_candidates) == 0:
                    continue

                results, exhausted_queries = self.get_goal_results(transformed_text_candidates)
                results = sorted(results, key=lambda x: -x.score)
                
                # Workaround for query budget bug in textattack
                if len(results) == 0:
                    continue

                if self.swap_threshold > cur_result.score - results[0].score:
                    cur_result = results[0]
                    # If the number of words changed, re-calculate the index order
                    if max(index_order) >= len(cur_result.attacked_text.words):
                        self.logger.log_result(i=i, result=cur_result)
                        return self.perform_search(cur_result)

                else:
                    continue

            self.logger.log_result(i=i, result=cur_result)

            i += 1

            # After traversing the input text, try again
            if i >= len(index_order):
                return self.perform_search(cur_result)
        return cur_result
