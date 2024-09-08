import torch
import textattack
from search_methods.pez_gradient_search import PEZGradientSearch
from transformations.nop import NOP
from goal_functions.increase_confidence import IncreaseConfidence
from utils.utils import get_model_wrapper, run_attack, random_sentence


def initialize_prompt(token_embedding, prompt_len, device):
    # randomly optimize prompt embeddings
    vocab_size = token_embedding.weight.data.shape[0]
    prompt_ids = torch.randint(vocab_size, (1, prompt_len)).to(device)
    prompt_embeds = token_embedding(prompt_ids).detach()

    return prompt_embeds

def unbounded_drift(model_name):
    model_wrapper = get_model_wrapper(model_name)

    # Construct our four components for `Attack`
    goal_function = IncreaseConfidence(model_wrapper, query_budget=30, eps=0.01)
    constraints = []
    transformation = NOP()
    search_method = PEZGradientSearch(model_wrapper, lr=0.1, wd=1, debug=True)

    # Construct the actual attack
    attack = textattack.Attack(goal_function, constraints, transformation, search_method)

    run_attack(attack=attack)

if __name__ == '__main__':
    unbounded_drift("mnoukhov/gpt2-imdb-sentiment-classifier")
