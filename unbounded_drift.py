import torch
import textattack
from search_methods.pez_gradient_search import PEZGradientSearch
from transformations.nop import NOP
from goal_functions.increase_confidence import IncreaseConfidenceUntargeted, IncreaseConfidenceTargeted
from utils.utils import get_model_wrapper, run_attack


def initialize_prompt(token_embedding, prompt_len, device):
    # randomly optimize prompt embeddings
    vocab_size = token_embedding.weight.data.shape[0]
    prompt_ids = torch.randint(vocab_size, (1, prompt_len)).to(device)
    prompt_embeds = token_embedding(prompt_ids).detach()

    return prompt_embeds

def unbounded_drift(model_name, targeted=False, max_iter=100, filter_by_target_class=False, target_class=0):
    model_wrapper = get_model_wrapper(model_name)

    # Construct our four components for `Attack`
    if targeted:
        goal_function = IncreaseConfidenceTargeted(model_wrapper, query_budget=max_iter, target_class=target_class)
    else:
        goal_function = IncreaseConfidenceUntargeted(model_wrapper, query_budget=max_iter)
    constraints = []
    transformation = NOP()
    search_method = PEZGradientSearch(model_wrapper, target_class=target_class, lr=0.4, wd=0, debug=True, max_iter=max_iter, filter_by_target_class=filter_by_target_class)

    # Construct the actual attack
    attack = textattack.Attack(goal_function, constraints, transformation, search_method)

    run_attack(attack=attack)

if __name__ == '__main__':
    #unbounded_drift("mnoukhov/gpt2-imdb-sentiment-classifier")
    unbounded_drift("finiteautomata/bertweet-base-sentiment-analysis", targeted=True, max_iter=5000, filter_by_target_class=True, target_class=2)
    unbounded_drift("cardiffnlp/twitter-roberta-base-sentiment-latest", targeted=True, max_iter=5000, filter_by_target_class=True, target_class=2)
