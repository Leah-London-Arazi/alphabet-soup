""""
NOTE: Since TextAttack transformations are on words, after each transformation we had
to decode the tokens, and then we encode them again.
By doing so, we may cause unnecessary changes in the tokens. For example, [3, 6]->"bead"->[37]
"""

import torch
from textattack.shared import AttackedText
from textattack.transformations import Transformation
from utils.models import get_grad_wrt_func, get_token_ids_without_special_tokens
from textattack.shared.utils import device as ta_device


class RandomTokenGradientBasedSwap(Transformation):
    def __init__(self, model_wrapper, top_n=1, target_class=None):
        self.model_wrapper = model_wrapper

        self.model = self.model_wrapper.model.to(device=ta_device)
        self.tokenizer = self.model_wrapper.tokenizer

        if not hasattr(self.model, "get_input_embeddings"):
            raise ValueError(
                "Model needs word embedding matrix for gradient-based word swap"
            )

        self.top_n = top_n
        self.is_black_box = False
        self.target_class = target_class
        self.token_ids = get_token_ids_without_special_tokens(self.tokenizer, device=ta_device)
        self.token_embeddings = self.model.get_input_embeddings().to(device=ta_device)

    def _get_replacement_words_by_grad(self, attacked_text, indices_to_replace):
        lookup_table = self.token_embeddings(self.token_ids).to(device=ta_device)
        self.input_ids = self.tokenizer(attacked_text.tokenizer_input,
                                        add_special_tokens=False,
                                        return_tensors="pt",
                                        padding=True,
                                        truncation=True).input_ids.to(device=ta_device)
        num_tokens_in_text = self.input_ids.shape[1]
        if self.target_class is not None:
            grad_output = get_grad_wrt_func(model_wrapper=self.model_wrapper,
                                            input_ids=self.input_ids,
                                            label=self.target_class)
        else:
            grad_output = self.model_wrapper.get_grad(attacked_text.tokenizer_input)
        emb_grad = grad_output["gradient"].clone().detach().to(device=ta_device)

        # grad differences between all tokens and original tokens
        diffs = torch.zeros(num_tokens_in_text, len(self.token_ids))

        for token_pos_in_text in range(num_tokens_in_text):
            b_grads = lookup_table @ emb_grad[token_pos_in_text]
            a_grad = b_grads[self.input_ids.squeeze(0)[token_pos_in_text]]
            diffs[token_pos_in_text] = b_grads - a_grad

        # Find best indices within 2-d tensor by flattening.
        token_idxs_sorted_by_grad = (-diffs).flatten().argsort()

        candidates = []
        for idx in token_idxs_sorted_by_grad.tolist()[:self.top_n]:
            token_pos_in_txt = idx // diffs.shape[1]
            token_id_pos_in_lookup = idx % diffs.shape[1]
            new_token_id = self.token_ids[token_id_pos_in_lookup].item()
            candidates.append((new_token_id, token_pos_in_txt))

        return candidates


    def _get_transformations(self, current_text, indices_to_replace):
        transformations = []
        for token_id, idx_in_tokenized_sentence in self._get_replacement_words_by_grad(
            current_text, indices_to_replace
        ):
            text_ids = self.input_ids.squeeze().clone()
            text_ids[idx_in_tokenized_sentence] = token_id
            transformed_attacked_text = AttackedText(text_input=self.tokenizer.decode(token_ids=text_ids, skip_special_tokens=True))
            transformations.append(transformed_attacked_text)
        return transformations
