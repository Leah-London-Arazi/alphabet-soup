"""
Reimplementation of Greedy Word Swap with Word Importance Ranking
"""
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import GreedyWordSwapWIR


class GreedyWordSwapThresholdWIR(GreedyWordSwapWIR):
    def __init__(self, wir_method="unk", unk_token="[UNK]", swap_threshold=0.0, num_transformations_per_word=1,
                 debug=False):
        super().__init__(wir_method=wir_method, unk_token=unk_token)
        self.swap_threshold = swap_threshold
        self.num_transformation_per_word = num_transformations_per_word
        self.debug = debug

    def perform_search(self, initial_result):
        attacked_text = initial_result.attacked_text
        index_order, exhausted_queries = self._get_index_order(attacked_text)
        cur_result = initial_result

        i = 0

        if self.debug:
            print(f"initial_result: {cur_result}")

        while i < len(index_order) and not exhausted_queries and cur_result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
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

                if results[0].score > cur_result.score - self.swap_threshold:
                    cur_result = results[0]
                else:
                    continue
            i += 1
            if self.debug:
                print(f"cur_result: {cur_result}")

        return cur_result
