###########
# Adapted from https://github.com/YuxinWenRick/hard-prompts-made-easy/blob/main/optim_utils.py
###########

import torch
from sentence_transformers.util import normalize_embeddings, semantic_search, dot_score
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared import AttackedText
from textattack.shared.utils import device as ta_device
from utils.attack import (get_filtered_token_ids_by_glove_score,
                          get_grad_wrt_func,
                          get_filtered_token_ids_by_bert_score,
                          get_filtered_token_ids__multi_prefix)
from utils.defaults import DEFAULT_CACHE_DIR, DEFAULT_PREFIXES
from utils.utils import create_cache_dir


class PEZGradientSearch(SearchMethod):
    def __init__(self,
                 model_wrapper,
                 lr,
                 wd,
                 target_class,
                 cache_dir=DEFAULT_CACHE_DIR,
                 max_iter=50,
                 filter_by_target_class=False,
                 filter_by_bert_score=False,
                 filter_by_glove_score=False,
                 debug=False):
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
        self.filter_by_bert_score = filter_by_bert_score
        self.filter_by_glove_score = filter_by_glove_score

        self.cache_dir = cache_dir
        create_cache_dir(self.cache_dir)

        self.debug = debug

    def perform_search(self, initial_result):
        if self.debug:
            print(f"initial_result: {initial_result}")
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
            token_ids = get_filtered_token_ids__multi_prefix(model=self.model,
                                                             tokenizer=self.tokenizer,
                                                             target_class=self.target_class,
                                                             confidence_threshold=0.5,
                                                             cache_dir=self.cache_dir,
                                                             prefixes=DEFAULT_PREFIXES,
                                                             debug=self.debug).to(device=ta_device)

        elif self.filter_by_bert_score:
            token_ids = get_filtered_token_ids_by_bert_score(tokenizer=self.tokenizer,
                                                 word_refs=["this is food",
                                                            "this is a type of meat",
                                                            "this is a type of cheese",
                                                            "this is a drink"],
                                                 score_threshold=0.8, debug=self.debug)

        elif self.filter_by_glove_score:
            token_ids = get_filtered_token_ids_by_glove_score(self.tokenizer, word_refs=["bitch"],
                                                              score_threshold=0.6, debug=self.debug)

        filtered_embedding_matrix = normalize_embeddings(self.token_embeddings(token_ids))

        # begin loop
        cur_result = initial_result
        i = 0
        exhausted_queries = False

        # PEZ optimization algorithm
        while (i < self.max_iter 
               and not exhausted_queries 
               and cur_result.goal_status != GoalFunctionResultStatus.SUCCEEDED):
            nn_indices = PEZGradientSearch._nn_project(prompt_embeds, filtered_embedding_matrix, token_ids)

            prompt_len = prompt_embeds.shape[0]
            nn_indices_grad = get_grad_wrt_func(self.model_wrapper,
                                                nn_indices.unsqueeze(0),
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


    @staticmethod
    def _nn_project(prompt_embeds, filtered_embedding_matrix, filtered_token_ids):
        with torch.no_grad():
            # Using the sentence transformers semantic search which is
            # a dot product exact kNN search between a set of
            # query vectors and a corpus of vectors

            prompt_embeds = normalize_embeddings(prompt_embeds)

            hits = semantic_search(prompt_embeds, filtered_embedding_matrix,
                                   query_chunk_size=prompt_embeds.shape[0],
                                   top_k=1,
                                   score_function=dot_score)

            nn_indices = torch.tensor([filtered_token_ids[hit[0]["corpus_id"]] for hit in hits],
                                      device=ta_device)

        return nn_indices
