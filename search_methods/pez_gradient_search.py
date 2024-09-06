###########
# Adapted from https://github.com/YuxinWenRick/hard-prompts-made-easy/blob/main/optim_utils.py
###########

from textattack.search_methods import SearchMethod
import torch
from sentence_transformers.util import normalize_embeddings, semantic_search, dot_score
from textattack.shared import AttackedText
from textattack.shared.utils import device as ta_device

class PEZGradientSearch(SearchMethod):
    def __init__(self, model_wrapper, lr, wd, max_iter=50, debug=False):
        # Unwrap model wrappers. Need raw model for gradient.
        self.model = model_wrapper.model

        if not hasattr(self.model, "get_input_embeddings"):
            raise ValueError(
                "Model needs word embedding matrix for gradient-based word swap"
            )

        self.model_wrapper = model_wrapper
        self.tokenizer = self.model_wrapper.tokenizer
        self.token_embeddings = self.model.get_input_embeddings()
        self.lr = lr
        self.wd = wd
        self.max_iter = max_iter
        self.debug = debug
        self.device = ta_device

    def perform_search(self, initial_result):
        attacked_text = initial_result.attacked_text

        # init
        text_ids = self.tokenizer(attacked_text.tokenizer_input, return_tensors='pt')["input_ids"]
        prompt_embeds = self.token_embeddings(text_ids).squeeze().detach().to(self.device)
        optimizer = torch.optim.AdamW([prompt_embeds], lr=self.lr, weight_decay=self.wd)

        # begin loop
        cur_result = initial_result
        i = 0
        search_over = False

        while i < self.max_iter and not search_over:
            nn_indices = self._nn_project(prompt_embeds)
            modified_text = self.tokenizer.decode(nn_indices)
            prompt_len = prompt_embeds.shape[0]
            modified_text_grad = self.model_wrapper.get_grad(modified_text)['gradient']
            prompt_embeds.grad = torch.tensor(modified_text_grad[:prompt_len], device=self.device)

            optimizer.step()
            optimizer.zero_grad()

            results, search_over = self.get_goal_results([AttackedText(text_input=modified_text)])
            cur_result = results[0]

            i += 1

            if self.debug:
                print(f"iteration: {i}, cur_result: {cur_result}")

        return cur_result


    @property
    def is_black_box(self):
        return False


    def _nn_project(self, embeds):
        with torch.no_grad():
            # Using the sentence transformers semantic search which is
            # a dot product exact kNN search between a set of
            # query vectors and a corpus of vectors

            embeds = normalize_embeddings(embeds)
            embedding_matrix = normalize_embeddings(self.token_embeddings.weight.data)

            hits = semantic_search(embeds, embedding_matrix,
                                   query_chunk_size=embeds.shape[0],
                                   top_k=1,
                                   score_function=dot_score)

            nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=self.device)

        return nn_indices
