from pydantic import BaseModel, ConfigDict
from consts import FilterTokenIDsMethod


class AttackParams(BaseModel):
    model_config = ConfigDict(extra='forbid')

    @classmethod
    def _from_args(cls, attack_params):
        kwargs = {}
        for param in attack_params:
            key, value = param.split("=")
            kwargs[key] = value
        return cls(**kwargs)


class CharacterRouletteBlackBoxAttackParams(AttackParams):
    swap_threshold: float = 0.1
    num_transformations_per_word: int = 1


class CharacterRouletteWhiteBoxAttackParams(AttackParams):
    top_n: int = 1
    num_random_tokens: int = 500
    beam_width: int = 10


class PEZAttackParams(AttackParams):
    lr: float = 0.4
    filter_token_ids_method: FilterTokenIDsMethod
    word_refs: list[str] = []


class GCGAttackParams(AttackParams):
    max_retries_per_iter: int = 100
    top_k: int = 256
    beam_width: int = 1
    filter_token_ids_method: FilterTokenIDsMethod
    word_refs: list[str] = []
