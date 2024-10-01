from recipes.base import BaseAttackRecipe
from schemas import CharacterRouletteBlackBoxAttackParams, CharacterRouletteWhiteBoxAttackParams
from textattack.transformations import CompositeTransformation, WordSwapRandomCharacterSubstitution, \
    WordSwapRandomCharacterDeletion, WordSwapRandomCharacterInsertion, WordSwapNeighboringCharacterSwap
from search_methods.greedy_word_swap_threshold_wir import GreedyWordSwapThresholdWIR
from search_methods.beam_search import BeamSearch
from transformations.random_token_gradient_based_swap import RandomTokenGradientBasedSwap
from transformations.word_swap_random_word import WordSwapRandomWord


class CharacterRouletteBlackBox(BaseAttackRecipe):
    def __init__(self, attack_params: CharacterRouletteBlackBoxAttackParams, **kwargs):
        super().__init__(attack_params=attack_params, **kwargs)

    def get_search_method(self):
       return GreedyWordSwapThresholdWIR(swap_threshold=self.attack_params.swap_threshold,
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


class CharacterRouletteWhiteBox(BaseAttackRecipe):
    def __init__(self, attack_params: CharacterRouletteWhiteBoxAttackParams, **kwargs):
        super().__init__(attack_params=attack_params, **kwargs)

    def get_transformation(self):
        return RandomTokenGradientBasedSwap(self.model_wrapper,
                                            top_n=self.attack_params.top_n,
                                            target_class=self.target_class)
    def get_search_method(self):
        return BeamSearch(beam_width=self.attack_params.beam_width)
