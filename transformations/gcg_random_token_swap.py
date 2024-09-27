import torch
from textattack.shared import AttackedText
from textattack.transformations import Transformation
from utils.attack import get_grad_wrt_func, get_filtered_token_ids
from textattack.shared.utils import device as ta_device
from utils.defaults import DEFAULT_CACHE_DIR
from utils.utils import create_dir


class GCGRandomTokenSwap(Transformation):
    def __init__(self,
                 model_wrapper,
                 goal_function,
                 max_retries_per_iter,
                 top_k,
                 word_refs,
                 filter_token_ids_method,
                 cache_dir=DEFAULT_CACHE_DIR,
                 debug=False):
        super().__init__()
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        self.model.to(device=ta_device)
        self.tokenizer = model_wrapper.tokenizer
        self.goal_function = goal_function
        self.target_class = self.goal_function.target_class

        self.max_retries_per_iter = max_retries_per_iter
        self.top_k = top_k

        self.cache_dir = cache_dir
        create_dir(self.cache_dir)

        # filter tokens using glove score
        self.token_embeddings = self.model.get_input_embeddings()
        self.debug = debug

        self.filter_token_ids_method = filter_token_ids_method
        self.token_ids = get_filtered_token_ids(filter_method=self.filter_token_ids_method,
                                                model=self.model,
                                                tokenizer=self.tokenizer,
                                                target_class=self.target_class,
                                                cache_dir=self.cache_dir,
                                                word_refs=word_refs,
                                                debug=self.debug)
    @property
    def is_black_box(self):
        return False

    def _sample_control(self, control_toks, loss_change_estimate):
        # Identify V_cand from AutoPrompt
        top_k = min(self.top_k, loss_change_estimate.shape[1])
        top_indices = (-loss_change_estimate).topk(top_k, dim=1).indices
        new_token_pos = torch.randint(low=0, high=len(control_toks), size=(1,)).item()
        new_token_idx = torch.randint(0, top_k, size=(1,)).item()
        new_token_val = self.token_ids[top_indices[new_token_pos][new_token_idx]]
        new_control_toks = control_toks.clone()
        new_control_toks[new_token_pos] = new_token_val
        return new_control_toks


    def _get_new_tokens_gcg(self, attacked_text):
        input_ids = self.tokenizer(attacked_text.tokenizer_input,
                                   add_special_tokens=False,
                                   return_tensors="pt",
                                   padding=True,
                                   truncation=True).input_ids.to(device=ta_device)

        logits = self.model(input_ids=input_ids).logits
        curr_score = self.goal_function._get_score(logits.squeeze(0), None)

        grad = get_grad_wrt_func(self.model_wrapper, input_ids, self.target_class)['gradient'].to(device=ta_device)
        grad = grad / grad.norm(dim=-1, keepdim=True)

        loss_change_estimate = grad @ self.token_embeddings(self.token_ids).T

        for _ in range(self.max_retries_per_iter):
            new_input_ids = self._sample_control(input_ids.squeeze(0), loss_change_estimate)

            # check if the replacement is better than the original
            logits = self.model(input_ids=new_input_ids.unsqueeze(0)).logits
            new_score = self.goal_function._get_score(logits.squeeze(0), None)
            if new_score > curr_score:
                return new_input_ids

        raise Exception("Max retries exceeded")


    def _get_transformations(self, current_text, indices_to_replace):
        new_tokens = self._get_new_tokens_gcg(current_text)
        transformations = [AttackedText(text_input=self.tokenizer.decode(token_ids=new_tokens))]
        
        return transformations
