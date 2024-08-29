import textattack
import transformers
from character_roullete.transformations.word_swap_random_unknown_word import WordSwapRandomUnknownWord
from character_roullete.character_roullete import IncreaseConfidence
from character_roullete.search_methods.greedy_word_swap_threshold_wir import GreedyWordSwapThresholdWIR

if __name__ == '__main__':
    # Load model, tokenizer, and model_wrapper
    model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
    tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

    # Construct our four components for `Attack`
    goal_function = IncreaseConfidence(model_wrapper)
    constraints = []
    transformation = WordSwapRandomUnknownWord(model_wrapper)
    search_method = GreedyWordSwapThresholdWIR(swap_threshold=0.05, debug=True)
    # Construct the actual attack
    attack = textattack.Attack(goal_function, constraints, transformation, search_method)
    input_text = "The cat sat on the mat."
    label = 1  # Positive
    attack_result = attack.attack(input_text, label)
    print(attack_result)
