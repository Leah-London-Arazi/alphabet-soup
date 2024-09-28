from pydantic import BaseModel

from schemas import CharacterRouletteBlackBoxAttackParams, CharacterRouletteWhiteBoxAttackParams, PEZAttackParams, GCGAttackParams
from utils.utils import disable_warnings
disable_warnings()

import textattack
from textattack.transformations import CompositeTransformation, WordSwapRandomCharacterSubstitution, \
    WordSwapRandomCharacterDeletion, WordSwapRandomCharacterInsertion, WordSwapNeighboringCharacterSwap

from search_methods.pez_gradient_search import PEZGradientSearch
from transformations.gcg_random_token_swap import GCGRandomTokenSwap
from goal_functions.increase_confidence import IncreaseConfidenceTargeted, IncreaseConfidenceUntargeted
from search_methods.greedy_word_swap_threshold_wir import GreedyWordSwapThresholdWIR
from search_methods.beam_search import BeamSearch
from transformations.nop import NOP
from transformations.random_token_gradient_based_swap import RandomTokenGradientBasedSwap
from transformations.word_swap_random_word import WordSwapRandomWord
from utils.attack import get_model_wrapper


class AlphabetSoupAttackRecipe:
    def __init__(self, model_name: str, targeted: bool, target_class: int, query_budget: int,
                 confidence_threshold: float, attack_params: BaseModel):
        self.model_name = model_name
        self.targeted = targeted
        self.target_class = target_class
        self.query_budget = query_budget
        self.confidence_threshold = confidence_threshold
        self.attack_params = attack_params

    @property
    def model_wrapper(self):
        return get_model_wrapper(self.model_name)

    def get_goal_function(self):
        if self.targeted:
            return IncreaseConfidenceTargeted(self.model_wrapper,
                                              target_class=self.target_class,
                                              query_budget=self.query_budget,
                                              threshold=self.confidence_threshold)
        return IncreaseConfidenceUntargeted(self.model_wrapper,
                                            query_budget=self.query_budget,
                                            threshold=self.confidence_threshold)

    def get_constraints(self):
        return []

    def get_transformation(self):
        return NOP()

    def get_search_method(self):
        return BeamSearch(beam_width=1)

    def get_attack(self):
        return textattack.Attack(self.get_goal_function(),
                                 self.get_constraints(),
                                 self.get_transformation(),
                                 self.get_search_method())

    @property
    def is_targeted_only(self):
        return False


class CharacterRouletteBlackBox(AlphabetSoupAttackRecipe):
    def __init__(self, attack_params: CharacterRouletteBlackBoxAttackParams, **kwargs):
        super().__init__(attack_params=attack_params, **kwargs)

    def get_search_method(self):
        GreedyWordSwapThresholdWIR(swap_threshold=self.attack_params.swap_threshold,
                                   num_transformations_per_word=self.attack_params.num_transformations_per_word)


class CharacterRouletteBlackBoxRandomChar(CharacterRouletteBlackBox):
    def get_transformation(self):
        return CompositeTransformation(
            [
                WordSwapRandomCharacterSubstitution(),
                WordSwapRandomCharacterDeletion(),
                WordSwapRandomCharacterInsertion(),
                WordSwapNeighboringCharacterSwap(),
            ]
        )


class CharacterRouletteBlackBoxRandomWord(CharacterRouletteBlackBox):
    def get_transformation(self):
        return WordSwapRandomWord()


class CharacterRouletteWhiteBox(AlphabetSoupAttackRecipe):
    def __init__(self, attack_params: CharacterRouletteWhiteBoxAttackParams, **kwargs):
        super().__init__(attack_params=attack_params, **kwargs)

    def get_transformation(self):
        return RandomTokenGradientBasedSwap(self.model_wrapper,
                                            top_n=self.attack_params.top_n,
                                            target_class=self.target_class)
    def get_search_method(self):
        return BeamSearch(beam_width=self.attack_params.beam_width)


class PEZ(AlphabetSoupAttackRecipe):
    def __init__(self, attack_params: PEZAttackParams, **kwargs):
        super().__init__(attack_params=attack_params, **kwargs)

    def get_search_method(self):
        return PEZGradientSearch(self.model_wrapper,
                                 target_class=self.target_class,
                                 lr=self.attack_params.lr,
                                 max_iter=self.query_budget,
                                 filter_token_ids_method=self.attack_params.filter_token_ids_method,
                                 word_refs=self.attack_params.word_refs,
                                 num_random_tokens=self.attack_params.num_random_tokens)

    @property
    def is_targeted_only(self):
        return True


class GCG(AlphabetSoupAttackRecipe):
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
                                  top_k=self.attack_params.top_k,
                                  num_random_tokens=self.attack_params.num_random_tokens,)
