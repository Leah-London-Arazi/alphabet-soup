import textattack
from textattack.transformations import WordSwapRandomCharacterSubstitution, WordSwapRandomCharacterDeletion, \
    WordSwapRandomCharacterInsertion
from transformations.word_swap_random_gradient_based import WordSwapTokenGradientBased
from transformations.word_swap_random_unknown_word import WordSwapRandomUnknownWord
from textattack.transformations.composite_transformation import CompositeTransformation
from textattack.search_methods import BeamSearch
from goal_functions.increase_confidence import IncreaseConfidence
from search_methods.greedy_word_swap_threshold_wir import GreedyWordSwapThresholdWIR
from utils.utils import get_model_wrapper, run_attack


def character_roulette_black_box__random_char(model_name):
    model_wrapper = get_model_wrapper(model_name)

    # Construct our four components for `Attack`
    goal_function = IncreaseConfidence(model_wrapper)
    constraints = []
    transformation = CompositeTransformation(
        [
            WordSwapRandomCharacterSubstitution(),
            WordSwapRandomCharacterDeletion(),
            WordSwapRandomCharacterInsertion(),
        ]
    )
    search_method = GreedyWordSwapThresholdWIR(swap_threshold=0.1, debug=True, num_transformations_per_word=3)

    # Construct the actual attack
    attack = textattack.Attack(goal_function, constraints, transformation, search_method)

    run_attack(attack=attack)


def character_roulette_black_box__random_word(model_name):
    model_wrapper = get_model_wrapper(model_name)

    # Construct our four components for `Attack`
    goal_function = IncreaseConfidence(model_wrapper)
    constraints = []
    transformation = WordSwapRandomUnknownWord()
    search_method = GreedyWordSwapThresholdWIR(swap_threshold=0.1, debug=True, num_transformations_per_word=3)

    # Construct the actual attack
    attack = textattack.Attack(goal_function, constraints, transformation, search_method)

    run_attack(attack=attack)


def character_roulette_white_box(model_name):
    model_wrapper = get_model_wrapper(model_name)

    # Construct our four components for `Attack`
    goal_function = IncreaseConfidence(model_wrapper, query_budget=30, eps=0.1)
    constraints = []
    transformation = WordSwapTokenGradientBased(model_wrapper, top_n=1, num_random_tokens=50)
    search_method = BeamSearch(beam_width=5)

    # Construct the actual attack
    attack = textattack.Attack(goal_function, constraints, transformation, search_method)

    run_attack(attack=attack, input_text="dknjks d,ddkls the alkdla.")


if __name__ == '__main__':
    character_roulette_black_box__random_char("mnoukhov/gpt2-imdb-sentiment-classifier")
    character_roulette_white_box(r"mnoukhov/gpt2-imdb-sentiment-classifier")
