from recipes.base import BaseAttackRecipe
from schemas import GCGAttackParams
from utils.utils import disable_warnings
disable_warnings()
from transformations.gcg_random_token_swap import GCGRandomTokenSwap
from search_methods.beam_search import BeamSearch


class GCG(BaseAttackRecipe):
    def __init__(self, attack_params: GCGAttackParams, **kwargs):
        super().__init__(attack_params=attack_params, **kwargs)

    def get_search_method(self):
        return BeamSearch(beam_width=1)

    def get_transformation(self):
        return GCGRandomTokenSwap(self.model_wrapper,
                                  goal_function=self.get_goal_function(),
                                  max_retries_per_iter=self.attack_params.max_retries_per_iter,
                                  filter_token_ids_method=self.attack_params.filter_token_ids_method,
                                  word_refs=self.attack_params.word_refs,
                                  score_threshold=self.attack_params.score_threshold,
                                  top_k=self.attack_params.top_k,
                                  num_random_tokens=self.attack_params.num_random_tokens,)
