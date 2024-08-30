"""
Word Swap by Gradient
-------------------------------
"""

import torch
from textattack.transformations.word_swaps.word_swap_gradient_based import WordSwapGradientBased
from textattack.models.wrappers import ModelWrapper


class WordSwapRandomGradientBased(WordSwapGradientBased):
    def __init__(self, model_wrapper, top_n=1):
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

    def _get_replacement_words_by_grad(self, attacked_text, indices_to_replace):
        lookup_table = self.model.get_input_embeddings().weight.data.cpu()

        grad_output = self.model_wrapper.get_grad(attacked_text.tokenizer_input)
        emb_grad = torch.tensor(grad_output["gradient"])
        text_ids = grad_output["ids"].squeeze()
        # grad differences between all flips and original word (eq. 1 from paper)
        vocab_size = lookup_table.size(0)
        diffs = torch.zeros(len(indices_to_replace), vocab_size)
        indices_to_replace = list(indices_to_replace)

        for j, word_idx in enumerate(indices_to_replace):
            # Make sure the word is in bounds.
            if word_idx >= len(emb_grad):
                continue
            # Get the grad w.r.t the one-hot index of the word.
            b_grads = lookup_table.mv(emb_grad[word_idx]).squeeze()
            a_grad = b_grads[text_ids[word_idx]]
            diffs[j] = b_grads - a_grad

        # Don't change to the pad token.
        diffs[:, self.tokenizer.pad_token_id] = float("-inf")

        # Find best indices within 2-d tensor by flattening.
        word_idxs_sorted_by_grad = (-diffs).flatten().argsort()

        candidates = []
        num_words_in_text, num_words_in_vocab = diffs.shape
        for idx in word_idxs_sorted_by_grad.tolist():
            idx_in_diffs = idx // num_words_in_vocab
            idx_in_vocab = idx % (num_words_in_vocab)
            idx_in_sentence = indices_to_replace[idx_in_diffs]
            word = self.tokenizer._convert_id_to_word(idx_in_vocab)
            candidates.append((word, idx_in_sentence))
            if len(candidates) == self.top_n:
                break

        return candidates
