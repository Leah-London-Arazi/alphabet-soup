###########
# Adapted from https://github.com/YuxinWenRick/hard-prompts-made-easy/blob/main/optim_utils.py
###########
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
import torch
from sentence_transformers.util import normalize_embeddings, semantic_search, dot_score
from textattack.shared import AttackedText
from textattack.shared.utils import device as ta_device
from utils import utils

DEFAULT_CACHE_DIR = "cache"

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

        self.debug = debug

    def perform_search(self, initial_result):
        attacked_text = initial_result.attacked_text

        # init
        text_ids = self.tokenizer(attacked_text.tokenizer_input, return_tensors='pt')["input_ids"].to(ta_device)
        prompt_embeds = self.token_embeddings(text_ids).squeeze().detach().to(ta_device)
        optimizer = torch.optim.AdamW([prompt_embeds], lr=self.lr, weight_decay=self.wd)
        token_ids = range(self.token_embeddings.num_embeddings)
        filtered_embedding_matrix = self.token_embeddings

        # filter embeddings based on classification confidence
        if self.filter_by_target_class:
            token_ids, filtered_embedding_matrix = utils.get_filtered_token_ids_multi_prefix(model=self.model,
                                                                                             tokenizer=self.tokenizer,
                                                                                             target_class=self.target_class,
                                                                                             confidence_threshold=0.5,
                                                                                             batch_size=60,
                                                                                             prefixes=["", "This is "],
                                                                                             cache_dir=self.cache_dir,
                                                                                             debug=self.debug)
        # begin loop
        cur_result = initial_result
        i = 0
        exhausted_queries = False

        # PEZ optimization algorithm
        while i < self.max_iter and not exhausted_queries and cur_result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
            nn_indices = self._nn_project(prompt_embeds, filtered_embedding_matrix, token_ids)
            modified_text = self.tokenizer.decode(nn_indices)
            prompt_len = prompt_embeds.shape[0]
            modified_text_grad = utils.get_grad_wrt_func(self.model_wrapper, modified_text,
                                                         label=self.target_class)['gradient']
            prompt_embeds.grad = torch.tensor(modified_text_grad[:prompt_len], device=ta_device)

            optimizer.step()
            optimizer.zero_grad()

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
