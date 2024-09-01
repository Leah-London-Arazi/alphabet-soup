"""
Word Swap Random by Gradient
-------------------------------
"""
import numpy as np
import torch
from textattack.shared import AttackedText
from textattack.transformations import WordSwap


class WordSwapTokenGradientBased(WordSwap):
    def __init__(self, model_wrapper, top_n=1, num_random_tokens=1):
        # Unwrap model wrappers. Need raw model for gradient.
        self.model = model_wrapper.model
        self.model_wrapper = model_wrapper
        self.tokenizer = self.model_wrapper.tokenizer
        # Make sure this model has all of the required properties.
        if not hasattr(self.model, "get_input_embeddings"):
            raise ValueError(
                "Model needs word embedding matrix for gradient-based word swap"
            )
        if not hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id:
            raise ValueError(
                "Tokenizer needs to have `pad_token_id` for gradient-based word swap"
            )

        self.top_n = top_n
        self.is_black_box = False
        self.num_random_tokens = num_random_tokens

    def _get_replacement_words_by_grad(self, attacked_text, indices_to_replace):
        lookup_table = self.model.get_input_embeddings().weight.data.cpu()
        grad_output = self.model_wrapper.get_grad(attacked_text.tokenizer_input)
        emb_grad = torch.tensor(grad_output["gradient"])
        text_ids = grad_output["ids"].squeeze()

        # grad differences between all tokens and original tokens
        tokens_indices_to_replace = self._get_tokens_indices_to_replace(attacked_text, indices_to_replace)
        diffs = torch.zeros(len(tokens_indices_to_replace), self.num_random_tokens)

        # sample tokens at random without self.tokenizer.pad_token_id
        random_token_idxes = np.random.randint(lookup_table.shape[0]-1, size=self.num_random_tokens)

        for j, token_idx in enumerate(tokens_indices_to_replace):
            b_grads = lookup_table.mv(emb_grad[token_idx]).squeeze()
            a_grad = b_grads[text_ids[token_idx]]
            diffs[j] = b_grads[random_token_idxes] - a_grad

        # Find best indices within 2-d tensor by flattening.
        token_idxs_sorted_by_grad = (-diffs).flatten().argsort()

        candidates = []
        for idx in token_idxs_sorted_by_grad.tolist():
            token_idx = idx // diffs.shape[0]
            new_token_id = random_token_idxes[idx % diffs.shape[1]]
            idx_in_tokenized_sentence = tokens_indices_to_replace[token_idx]
            candidates.append((new_token_id, idx_in_tokenized_sentence))
            if len(candidates) == self.top_n:
                break

        return candidates

    def _get_tokens_indices_to_replace(self, attacked_text, indices_to_replace):
        # TODO: check that the tokenizer returns "BatchEncoding"
        tokens_indices_to_replace = []
        batch_encoding = self.tokenizer(attacked_text.tokenizer_input)
        words_idxes = batch_encoding.word_ids()
        for i, words_idx in enumerate(words_idxes):
            if words_idx in indices_to_replace:
                tokens_indices_to_replace.append(i)

        return tokens_indices_to_replace

    def _get_transformations(self, attacked_text, indices_to_replace):
        """Returns a list of all possible transformations for `text`.

        If indices_to_replace is set, only replaces words at those
        indices.
        """
        transformations = []
        for token, idx_in_tokenized_sentence in self._get_replacement_words_by_grad(
            attacked_text, indices_to_replace
        ):
            text_ids = self.tokenizer(attacked_text.tokenizer_input)["input_ids"]
            text_ids[idx_in_tokenized_sentence] = token
            transformed_attacked_text = AttackedText(text_input=self.tokenizer.decode(token_ids=text_ids))
            transformations.append(transformed_attacked_text)
        return transformations
