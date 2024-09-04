import transformers
from textattack.models.wrappers import HuggingFaceModelWrapper 
import random
import string

def get_model_wrapper(model_name):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return HuggingFaceModelWrapper(model, tokenizer)


def print_perturbed_result(perturbed_result):
    print(f"perturbed text: '{perturbed_result.attacked_text.text}'.")
    print(f"classified as {perturbed_result.output} with score of {perturbed_result.score}.")
    print(f"used {perturbed_result.num_queries} queries.")


def run_attack(attack, input_text="The movie was filmed somewhere at some time.", label=1):
    attack_result = attack.attack(input_text, label)
    perturbed_result = attack_result.perturbed_result
    print_perturbed_result(perturbed_result)

def random_word(min_len=3, max_len=10):
    length = random.randint(min_len, max_len)  # Random word length
    word = ''.join(random.choices(string.printable, k=length))  # Generate a word
    return word