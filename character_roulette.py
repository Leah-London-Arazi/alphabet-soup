from utils.utils import disable_tensorflow_warnings
disable_tensorflow_warnings()

import textattack
from textattack.transformations import WordSwapRandomCharacterSubstitution, WordSwapRandomCharacterDeletion, \
    WordSwapRandomCharacterInsertion, WordSwapNeighboringCharacterSwap
from textattack.transformations.composite_transformation import CompositeTransformation
from textattack.search_methods import BeamSearch

from transformations.random_token_gradient_based_swap import RandomTokenGradientBasedSwap
from transformations.word_swap_random_word import WordSwapRandomWord
from goal_functions.increase_confidence import IncreaseConfidenceUntargeted, IncreaseConfidenceTargeted
from search_methods.greedy_word_swap_threshold_wir import GreedyWordSwapThresholdWIR
from utils.attack import get_model_wrapper, run_attack


def character_roulette_black_box__random_char(model_name, targeted=True, target_class=0, query_budget=500,
                                              threshold=0.9):
    model_wrapper = get_model_wrapper(model_name)

    if targeted:
        goal_function = IncreaseConfidenceTargeted(model_wrapper, target_class=target_class,
                                                   query_budget=query_budget, threshold=threshold)
    else:
        goal_function = IncreaseConfidenceUntargeted(model_wrapper, query_budget=query_budget,
                                                     threshold=threshold)
    constraints = []
    transformation = CompositeTransformation(
        [
            WordSwapRandomCharacterSubstitution(),
            WordSwapRandomCharacterDeletion(),
            WordSwapRandomCharacterInsertion(),
            WordSwapNeighboringCharacterSwap(),
        ]
    )
    search_method = GreedyWordSwapThresholdWIR(swap_threshold=0.05, debug=True, num_transformations_per_word=1)

    # Construct the actual attack
    attack = textattack.Attack(goal_function, constraints, transformation, search_method)

    run_attack(attack=attack)


def character_roulette_black_box__random_word(model_name, targeted=True, target_class=0, query_budget=500,
                                              threshold=0.9):
    model_wrapper = get_model_wrapper(model_name)

    if targeted:
        goal_function = IncreaseConfidenceTargeted(model_wrapper, target_class=target_class,
                                                   query_budget=query_budget, threshold=threshold)
    else:
        goal_function = IncreaseConfidenceUntargeted(model_wrapper, query_budget=query_budget,
                                                     threshold=threshold)
        
    constraints = []
    transformation = WordSwapRandomWord()
    search_method = GreedyWordSwapThresholdWIR(swap_threshold=0.1, debug=True, num_transformations_per_word=1)

    # Construct the actual attack
    attack = textattack.Attack(goal_function, constraints, transformation, search_method)

    run_attack(attack=attack)


def character_roulette_white_box(model_name, targeted=False, query_budget=30):
    model_wrapper = get_model_wrapper(model_name)

    target_class = 0

    # Construct our four components for `Attack`
    if targeted:
        goal_function = IncreaseConfidenceTargeted(model_wrapper, query_budget=query_budget, target_class=target_class)
    else:
        goal_function = IncreaseConfidenceUntargeted(model_wrapper, query_budget=query_budget, threshold=0.9)
    constraints = []
    transformation = RandomTokenGradientBasedSwap(model_wrapper, top_n=1, num_random_tokens=500, target_class=target_class)
    search_method = BeamSearch(beam_width=10)

    # Construct the actual attack
    attack = textattack.Attack(goal_function, constraints, transformation, search_method)

    run_attack(attack=attack)


if __name__ == '__main__':
    character_roulette_black_box__random_char("cardiffnlp/twitter-roberta-base-sentiment-latest")
    # character_roulette_black_box__random_word("cardiffnlp/twitter-roberta-base-sentiment-latest")
    # character_roulette_white_box("mnoukhov/gpt2-imdb-sentiment-classifier")
    # character_roulette_white_box("cardiffnlp/twitter-roberta-base-sentiment-latest", targeted=True, query_budget=100)
