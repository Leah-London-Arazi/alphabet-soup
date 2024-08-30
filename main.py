import textattack
import transformers
from textattack.transformations import WordSwapRandomCharacterSubstitution, WordSwapRandomCharacterDeletion, \
    WordSwapRandomCharacterInsertion
from transformations.word_swap_random_unknown_word import WordSwapRandomUnknownWord
from textattack.transformations.composite_transformation import CompositeTransformation
from goal_functions.increase_confidence import IncreaseConfidence
from search_methods.greedy_word_swap_threshold_wir import GreedyWordSwapThresholdWIR

def get_model_wrapper(model_name):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

def character_roulette_random_char(model_name):
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
    input_text = "The movie was filmed somewhere at some time."
    label = 1  # Positive
    attack_result = attack.attack(input_text, label)
    print(attack_result)

def character_roulette_random_word(model_name):
    model_wrapper = get_model_wrapper(model_name)
    # Construct our four components for `Attack`
    goal_function = IncreaseConfidence(model_wrapper)
    constraints = []
    transformation = WordSwapRandomUnknownWord()
    search_method = GreedyWordSwapThresholdWIR(swap_threshold=0.1, debug=True, num_transformations_per_word=3)

    # Construct the actual attack
    attack = textattack.Attack(goal_function, constraints, transformation, search_method)
    input_text = "The movie was filmed somewhere at some time."
    label = 1  # Positive
    attack_result = attack.attack(input_text, label)
    print(attack_result)


if __name__ == '__main__':
    character_roulette_random_char("mnoukhov/gpt2-imdb-sentiment-classifier")