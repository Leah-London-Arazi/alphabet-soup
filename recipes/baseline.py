from recipes.base import BaseAttackRecipe
from schemas import AttackParams


class Baseline(BaseAttackRecipe):
    def __init__(self, attack_params: AttackParams, **kwargs):
        super().__init__(**kwargs, attack_params=attack_params)

