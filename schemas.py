from typing import Optional
import typing
from pydantic import BaseModel, ConfigDict
from consts import FilterTokenIDsMethod


class AttackParams(BaseModel):
    model_config = ConfigDict(extra='forbid')

    @classmethod
    def _from_args(cls, attack_params):
        kwargs = {}
        for param in attack_params:
            key, value = param.split("=")
            if typing.get_origin(cls.model_fields[key].annotation) == list:
                value = value.split(",")
            kwargs[key] = value
        return cls(**kwargs)


class CharacterRouletteBlackBoxAttackParams(AttackParams):
    swap_threshold: float = 0.1
    num_transformations_per_word: int = 1


class CharacterRouletteWhiteBoxAttackParams(AttackParams):
    top_n: int = 3
    beam_width: int = 10


class PEZAttackParams(AttackParams):
    lr: float = 0.4
    filter_token_ids_method: Optional[FilterTokenIDsMethod] = None
    word_refs: list[str] = []
    num_random_tokens: int = 10


class GCGAttackParams(AttackParams):
    max_retries_per_iter: int = 100
    top_k: int = 256
    filter_token_ids_method: Optional[FilterTokenIDsMethod] = None
    word_refs: list[str] = []
    num_random_tokens: int = 10
