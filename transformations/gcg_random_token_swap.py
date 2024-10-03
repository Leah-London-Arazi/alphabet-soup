import torch
from textattack.shared import AttackedText
from textattack.transformations import Transformation
from utils.attack import get_grad_wrt_func, get_filtered_token_ids
from textattack.shared.utils import device as ta_device
from utils.defaults import DEFAULT_CACHE_DIR
from utils.utils import create_dir, get_logger


class GCGRandomTokenSwap(Transformation):
    def __init__(self,
                 model_wrapper,
                 goal_function,
                 n_samples_per_iter,
                 top_k,
                 word_refs,
                 score_threshold,
                 num_random_tokens,
                 filter_token_ids_method,
                 cache_dir=DEFAULT_CACHE_DIR):
        super().__init__()
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        self.model.to(device=ta_device)
        self.tokenizer = model_wrapper.tokenizer
        self.goal_function = goal_function

        self.target_class = self.goal_function.target_class

        self.n_samples_per_iter = n_samples_per_iter
        self.top_k = top_k

        self.cache_dir = cache_dir
        create_dir(self.cache_dir)

        # filter tokens using glove score
        self.token_embeddings = self.model.get_input_embeddings()
        self.logger = get_logger(self.__module__)

        self.filter_token_ids_method = filter_token_ids_method
        self.token_ids = get_filtered_token_ids(filter_method=self.filter_token_ids_method,
                                                model=self.model,
                                                tokenizer=self.tokenizer,
                                                target_class=self.target_class,
                                                cache_dir=self.cache_dir,
                                                word_refs=word_refs,
                                                score_threshold=score_threshold,
                                                num_random_tokens=num_random_tokens)

    @property
    def is_black_box(self):
        return False

    def _sample_control(self, control_toks, loss_change_estimate, n_samples):
        # Identify V_cand from AutoPrompt
        top_k = min(self.top_k, loss_change_estimate.shape[1])
        top_indices = (-loss_change_estimate).topk(top_k, dim=1).indices

        new_control_toks_list = []

        for _ in range(n_samples):
            new_token_pos = torch.randint(low=0, high=len(control_toks), size=(1,)).item()
            new_token_idx = torch.randint(0, top_k, size=(1,)).item()
            new_token_val = self.token_ids[top_indices[new_token_pos][new_token_idx]]
            new_control_toks = control_toks.clone()
            new_control_toks[new_token_pos] = new_token_val
            new_control_toks_list.append(new_control_toks)

        return new_control_toks_list


    def _get_new_tokens_gcg(self, attacked_text):
        input_ids = self.tokenizer(attacked_text.tokenizer_input,
                                   add_special_tokens=False,
                                   return_tensors="pt",
                                   padding=True,
                                   truncation=True).input_ids.to(device=ta_device)


        grad = get_grad_wrt_func(self.model_wrapper, input_ids, self.target_class)['gradient'].to(device=ta_device)
        grad = grad / grad.norm(dim=-1, keepdim=True)

        loss_change_estimate = grad @ self.token_embeddings(self.token_ids).T

        new_input_ids_list = self._sample_control(input_ids.squeeze(0), loss_change_estimate, self.n_samples_per_iter)

        return new_input_ids_list


    def _get_transformations(self, current_text, indices_to_replace):
        new_tokens_list = self._get_new_tokens_gcg(current_text)
        transformations = [AttackedText(text_input=self.tokenizer.decode(token_ids=new_tokens)) for new_tokens in new_tokens_list]
        
        return transformations
