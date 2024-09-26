from utils.utils import disable_tensorflow_warnings
disable_tensorflow_warnings()

import textattack
from textattack.search_methods import BeamSearch
from transformations.gcg_random_token_swap import GCGRandomTokenSwap
from goal_functions.increase_confidence import IncreaseConfidenceTargeted
from utils.attack import get_model_wrapper, run_attack


def gcg(model_name, target_class=0, max_iter=100, max_retries_per_iter=100, top_k=256,
        filter_by_target_class=False, filter_by_glove_score=False, debug=False):
    model_wrapper = get_model_wrapper(model_name)

    goal_function = IncreaseConfidenceTargeted(model_wrapper, query_budget=max_iter,
                                               target_class=target_class)
    constraints = []
    transformation = GCGRandomTokenSwap(model_wrapper, goal_function=goal_function,
                                        max_retries_per_iter=max_retries_per_iter, top_k=top_k,
                                        filter_by_target_class=filter_by_target_class,
                                        filter_by_glove_score=filter_by_glove_score, debug=debug)
    # Greedy search
    search_method = BeamSearch(beam_width=1)

    # Construct the actual attack
    attack = textattack.Attack(goal_function, constraints, transformation, search_method)

    run_attack(attack=attack)

if __name__ == '__main__':
    gcg("finiteautomata/bertweet-base-sentiment-analysis", target_class=0, filter_by_target_class=True, debug=True)
    gcg("cardiffnlp/twitter-roberta-base-sentiment-latest", target_class=0, filter_by_glove_score=True, debug=True)
