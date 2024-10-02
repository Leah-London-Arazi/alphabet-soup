from pydantic import BaseModel
from textattack import Attack
from goal_functions.increase_confidence import IncreaseConfidenceTargeted, IncreaseConfidenceUntargeted
from search_methods.beam_search import BeamSearch
from transformations.nop import NOP
from utils.attack import get_model_wrapper


class BaseAttackRecipe:
    def __init__(self, model_name: str, targeted: bool, target_class: int, query_budget: int,
                 confidence_threshold: float, attack_params: BaseModel):
        self.model_name = model_name
        self.targeted = targeted
        self.target_class = target_class
        self.query_budget = query_budget
        self.confidence_threshold = confidence_threshold
        self.attack_params = attack_params

    @property
    def model_wrapper(self):
        return get_model_wrapper(self.model_name)

    def get_goal_function(self):
        if self.targeted:
            return IncreaseConfidenceTargeted(self.model_wrapper,
                                              target_class=self.target_class,
                                              query_budget=self.query_budget,
                                              threshold=self.confidence_threshold)
        return IncreaseConfidenceUntargeted(self.model_wrapper,
                                            query_budget=self.query_budget,
                                            threshold=self.confidence_threshold)

    def get_constraints(self):
        return []

    def get_transformation(self):
        return NOP()

    def get_search_method(self):
        return BeamSearch(beam_width=1)

    def get_attack(self):
        return Attack(self.get_goal_function(),
                                 self.get_constraints(),
                                 self.get_transformation(),
                                 self.get_search_method())

    @property
    def is_targeted_only(self):
        return False
