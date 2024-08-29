import textattack
import transformers
from character_roullete.transformations.word_swap_random_unknown_word import WordSwapRandomUnknownWord
from character_roullete.character_roullete import IncreaseConfidence

if __name__ == '__main__':
    # Load model, tokenizer, and model_wrapper
    model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
    tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
    
    # Construct our four components for `Attack`
    from textattack.transformations import WordSwapEmbedding
    from textattack.search_methods import GreedyWordSwapWIR
    goal_function = IncreaseConfidence(model_wrapper)
    constraints =  []
    transformation = WordSwapRandomUnknownWord(model_wrapper)
    search_method = GreedyWordSwapWIR()
    # Construct the actual attack
    attack = textattack.Attack(goal_function, constraints, transformation, search_method)
    input_text = "I really enjoyed the new movie that came out last month."
    label = 1 #Positive
    attack_result = attack.attack(input_text, label)
