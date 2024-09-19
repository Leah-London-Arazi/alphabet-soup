import torch
from textattack.shared import AttackedText
from textattack.transformations import WordSwap
from utils import utils
from textattack.shared.utils import device as ta_device


class GCGRandomTokenSwap(WordSwap):
    def __init__(self, model_wrapper, goal_function, max_retries_per_iter, top_k):
        super().__init__()
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        self.model.to(device=ta_device)
        self.tokenizer = model_wrapper.tokenizer
        self.goal_function = goal_function

        self.max_retries_per_iter = max_retries_per_iter
        self.top_k = top_k

        self.is_black_box = False


    def _sample_control(self, control_toks, grad):
        top_indices = (-grad).topk(self.top_k, dim=1).indices
        new_token_pos = torch.randint(low=0, high=len(control_toks), size=(1,)).item()
        new_token_idx = torch.randint(0, self.top_k, size=(1,)).item()
        new_token_val = top_indices[new_token_pos][new_token_idx]
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

        grad = utils.get_grad_wrt_func(self.model_wrapper, input_ids, self.goal_function.target_class)['gradient']
        grad = grad / grad.norm(dim=-1, keepdim=True)

        for _ in range(self.max_retries_per_iter):
            new_input_ids = self._sample_control(input_ids.squeeze(0), grad)

            # check if the replacement is better than the original
            logits = self.model(input_ids=new_input_ids.unsqueeze(0)).logits
            new_score = self.goal_function._get_score(logits.squeeze(0), None)
            if new_score > curr_score:
                return new_input_ids

        raise Exception("Max retries exceeded")


    def _get_transformations(self, current_text, indices_to_replace):
        # TODO: add the attacked text to transformations in case all transformations reduced accuracy
        new_tokens = self._get_new_tokens_gcg(current_text)
        transformations = [AttackedText(text_input=self.tokenizer.decode(token_ids=new_tokens))]
        
        return transformations
