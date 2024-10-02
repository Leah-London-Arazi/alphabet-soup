from recipes.base import BaseAttackRecipe
from schemas import PEZAttackParams
from search_methods.pez_gradient_search import PEZGradientSearch


class PEZ(BaseAttackRecipe):
    def __init__(self, attack_params: PEZAttackParams, **kwargs):
        super().__init__(attack_params=attack_params, **kwargs)

    def get_search_method(self):
        return PEZGradientSearch(self.model_wrapper,
                                 target_class=self.target_class,
                                 lr=self.attack_params.lr,
                                 max_iter=self.query_budget,
                                 filter_token_ids_method=self.attack_params.filter_token_ids_method,
                                 word_refs=self.attack_params.word_refs,
                                 score_threshold=self.attack_params.score_threshold,
                                 num_random_tokens=self.attack_params.num_random_tokens)

    @property
    def is_targeted_only(self):
        return True
