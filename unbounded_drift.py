from utils.utils import disable_tensorflow_warnings
disable_tensorflow_warnings()

import torch
import textattack
from search_methods.pez_gradient_search import PEZGradientSearch
from transformations.nop import NOP
from goal_functions.increase_confidence import IncreaseConfidenceTargeted
from utils.attack import get_model_wrapper, run_attack


def initialize_prompt(token_embedding, prompt_len, device):
    # randomly optimize prompt embeddings
    vocab_size = token_embedding.weight.data.shape[0]
    prompt_ids = torch.randint(vocab_size, (1, prompt_len)).to(device)
    prompt_embeds = token_embedding(prompt_ids).detach()

    return prompt_embeds


def unbounded_drift(model_name, max_iter=100, target_class=0, debug=False):
    model_wrapper = get_model_wrapper(model_name)

    # Construct our four components for `Attack`
    goal_function = IncreaseConfidenceTargeted(model_wrapper, query_budget=max_iter, target_class=target_class)
    constraints = []
    transformation = NOP()
    search_method = PEZGradientSearch(model_wrapper,
                                      target_class=goal_function.target_class,
                                      lr=0.4,
                                      wd=0,
                                      debug=debug,
                                      max_iter=max_iter)

    # Construct the actual attack
    attack = textattack.Attack(goal_function, constraints, transformation, search_method)

    run_attack(attack=attack)


def bounded_drift(model_name, max_iter=100, target_class=0, filter_by_target_class=False,
                  filter_by_bert_score=False, filter_by_glove_score=False, lr=0.4, wd=0, debug=False):
    model_wrapper = get_model_wrapper(model_name)

    # Construct our four components for `Attack`
    goal_function = IncreaseConfidenceTargeted(model_wrapper, query_budget=max_iter,
                                               target_class=target_class, threshold=0.8)
    constraints = []
    transformation = NOP()
    search_method = PEZGradientSearch(model_wrapper, target_class=goal_function.target_class, lr=lr,
                                      wd=wd, debug=debug, max_iter=max_iter,
                                      filter_by_target_class=filter_by_target_class,
                                      filter_by_bert_score=filter_by_bert_score,
                                      filter_by_glove_score=filter_by_glove_score)

    # Construct the actual attack
    attack = textattack.Attack(goal_function, constraints, transformation, search_method)

    run_attack(attack=attack)


if __name__ == '__main__':
    unbounded_drift("mnoukhov/gpt2-imdb-sentiment-classifier")
    bounded_drift("finiteautomata/bertweet-base-sentiment-analysis", max_iter=100,
                  filter_by_target_class=True, target_class=2, debug=True)
    # by target class
    bounded_drift("cardiffnlp/twitter-roberta-base-sentiment-latest", max_iter=100,
                  filter_by_target_class=True, target_class=2, debug=True)
    # glove
    bounded_drift("cardiffnlp/twitter-roberta-base-sentiment-latest", max_iter=500,
                  filter_by_glove_score=True, target_class=0, lr=0.05, debug=True)
