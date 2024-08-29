"""

CharacterRoulette
========================================
(Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers)

"""

from textattack import Attack
from textattack.constraints.overlap import LevenshteinEditDistance
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import (
    CompositeTransformation,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterSubstitution,
)

from textattack.goal_functions import ClassificationGoalFunction


class IncreaseConfidence(ClassificationGoalFunction):
    def __init__(self, *args, eps=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps

    def _get_score(self, model_output, _):
        return model_output.max()

    def _is_goal_complete(self, model_output, _):
        return 1 - model_output.max() <= self.eps


def character_roulette_black_box(model):
    transformation = CompositeTransformation(
        [
            WordSwapNeighboringCharacterSwap(),
            WordSwapRandomCharacterSubstitution(),
            WordSwapRandomCharacterDeletion(),
            WordSwapRandomCharacterInsertion(),
        ]
    )

    constraints = [LevenshteinEditDistance(30)]
    goal_function = IncreaseConfidence(model)
    search_method = GreedyWordSwapWIR()

    return Attack(goal_function, constraints, transformation, search_method)
