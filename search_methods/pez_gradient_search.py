###########
# Adapted from https://github.com/YuxinWenRick/hard-prompts-made-easy/blob/main/optim_utils.py
###########
import os
from pathlib import Path

import numpy as np
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
import torch
from sentence_transformers.util import normalize_embeddings, semantic_search, dot_score
from textattack.shared import AttackedText
from textattack.shared.utils import device as ta_device
from utils import utils

DEFAULT_CACHE_DIR = "cache"
BATCH_SIZE = 1024

class PEZGradientSearch(SearchMethod):
    def __init__(self, model_wrapper, lr, wd, target_class, cache_dir=None, max_iter=50, filter_by_target_class=False, debug=False):
        # Unwrap model wrappers. Need raw model for gradient.
        self.model = model_wrapper.model

        if not hasattr(self.model, "get_input_embeddings"):
            raise ValueError(
                "Model needs word embedding matrix for gradient-based word swap"
            )

        self.model.eval()
        self.model.to(ta_device)

        self.model_wrapper = model_wrapper
        self.tokenizer = self.model_wrapper.tokenizer
        self.token_embeddings = self.model.get_input_embeddings()

        self.lr = lr
        self.wd = wd
        self.max_iter = max_iter

        self.target_class = target_class
        self.filter_by_target_class = filter_by_target_class

        if cache_dir is None:
            self.cache_dir = DEFAULT_CACHE_DIR

        cache_directory = Path(self.cache_dir)
        cache_directory.mkdir(parents=True, exist_ok=True)

        self.debug = debug

    def perform_search(self, initial_result):
        # we optimize the tokens directly so we may receive an "irreversible" sequence of tokens,
        # meaning, after decoding and encoding it again the tokens would not restore.

        attacked_text = initial_result.attacked_text

        # init
        text_ids = self.tokenizer(attacked_text.tokenizer_input, return_tensors='pt')["input_ids"].to(ta_device)
        prompt_embeds = self.token_embeddings(text_ids).squeeze().detach().to(ta_device)
        optimizer = torch.optim.AdamW([prompt_embeds], lr=self.lr, weight_decay=self.wd)
        token_ids = torch.tensor(range(self.token_embeddings.num_embeddings), device=ta_device)

        # filter embeddings based on classification confidence
        if self.filter_by_target_class:
            token_ids = self._get_filtered_token_ids__multi_prefix(confidence_threshold=0.5,
                                                                   batch_size=BATCH_SIZE,
                                                                   prefixes=["", "This is "]).to(device=ta_device)


        filtered_embedding_matrix = normalize_embeddings(self.token_embeddings(token_ids))

        # begin loop
        cur_result = initial_result
        i = 0
        exhausted_queries = False

        # PEZ optimization algorithm
        while i < self.max_iter and not exhausted_queries and cur_result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
            nn_indices = self._nn_project(prompt_embeds, filtered_embedding_matrix, token_ids)

            prompt_len = prompt_embeds.shape[0]
            nn_indices_grad = utils.get_grad_wrt_func(self.model_wrapper, nn_indices.unsqueeze(0),
                                                         label=self.target_class)['gradient']
            prompt_embeds.grad = torch.tensor(nn_indices_grad[:prompt_len], device=ta_device)

            optimizer.step()
            optimizer.zero_grad()

            modified_text = self.tokenizer.decode(nn_indices)
            results, exhausted_queries = self.get_goal_results([AttackedText(text_input=modified_text)])
            cur_result = results[0]

            i += 1

            if self.debug:
                print(f"iteration: {i}, cur_result: {cur_result}")

        return cur_result


    @property
    def is_black_box(self):
        return False


    def _nn_project(self, prompt_embeds, filtered_embedding_matrix, filtered_token_ids):
        with torch.no_grad():
            # Using the sentence transformers semantic search which is
            # a dot product exact kNN search between a set of
            # query vectors and a corpus of vectors

            prompt_embeds = normalize_embeddings(prompt_embeds)

            hits = semantic_search(prompt_embeds, filtered_embedding_matrix,
                                   query_chunk_size=prompt_embeds.shape[0],
                                   top_k=1,
                                   score_function=dot_score)

            nn_indices = torch.tensor([filtered_token_ids[hit[0]["corpus_id"]] for hit in hits], device=ta_device)

        return nn_indices


    def _get_filtered_token_ids__multi_prefix(self, confidence_threshold, prefixes, batch_size):
        # filter embeddings based on classification confidence
        all_token_ids = list(range(len(self.tokenizer)))
        token_ids = torch.tensor(all_token_ids).cpu()

        for prefix in prefixes:
            cache_file_name = f"model={self.model.name_or_path.replace("/", "_")}_target_class={self.target_class}_confidence_threshold={confidence_threshold}_prefix={prefix}.pt"
            cache_file_path = os.path.join(self.cache_dir, cache_file_name)

            if os.path.exists(cache_file_path):
                token_ids_prefix = torch.load(cache_file_path)

            else:
                token_ids_prefix = utils.get_filtered_token_ids(model=self.model,
                                                                tokenizer=self.tokenizer,
                                                                target_class=self.target_class,
                                                                confidence_threshold=confidence_threshold,
                                                                batch_size=batch_size,
                                                                prefix=prefix)
                torch.save(token_ids_prefix, cache_file_path)

            token_ids = torch.tensor(np.intersect1d(token_ids_prefix.cpu(), token_ids))

        if token_ids.shape[0] == 0:
            raise Exception("Filtered all tokens!")

        if self.debug:
            print(f"{len(token_ids)} tokens remaining after filtering")
            filtered_tokens = torch.tensor(np.setdiff1d(all_token_ids, token_ids))
            filtered_words = self.tokenizer.batch_decode([filtered_tokens])
            print(f"Filtered the following tokens: {filtered_words}")

        return token_ids
