# implemented GCG algorithm from https://arxiv.org/pdf/2307.15043

import torch
import transformers
from textattack.shared.utils import device as ta_device
import utils.utils
from textattack.models.wrappers import HuggingFaceModelWrapper

def sample_control(control_toks, grad, topk=256):
    top_indices = (-grad).topk(topk, dim=1).indices
    new_token_pos = torch.randint(low=0, high=len(control_toks), size=(1,)).item()
    new_token_idx = torch.randint(0, topk, size=(1,)).item()
    new_token_val = top_indices[new_token_pos][new_token_idx]
    new_control_toks = control_toks.clone()
    new_control_toks[new_token_pos] = new_token_val
    return new_control_toks


def gcg_step(model, tokenizer, input_ids, top_k, target_label, retries=100):
    curr_score = torch.nn.functional.softmax(model(input_ids=input_ids).logits, dim=1)[0][target_label]
    grad = utils.utils.get_grad_wrt_func(HuggingFaceModelWrapper(model, tokenizer), input_ids, target_label)['gradient']
    grad = grad / grad.norm(dim=-1, keepdim=True)
    for i in range(retries):
        # check if the replacement is better from the original
        new_input_ids = sample_control(input_ids.squeeze(0), grad, top_k)
        new_score = torch.nn.functional.softmax(model(input_ids=new_input_ids.unsqueeze(0)).logits, dim=1)[0][target_label]
        if new_score > curr_score:
            return new_input_ids

    raise Exception("max retries reached")


def gcg(model, tokenizer, initial_text, top_k, num_iter, target_label, debug=False):
    model.to(device=ta_device)
    input_ids = tokenizer(initial_text,
                          add_special_tokens=True,
                          return_tensors="pt",
                          padding=True,
                          truncation=True).input_ids.to(device=ta_device)

    for i in range(num_iter):
        input_ids = gcg_step(model, tokenizer, input_ids, top_k, target_label)
        input_ids = input_ids.unsqueeze(0)

        score = torch.nn.functional.softmax(model(input_ids=input_ids).logits, dim=1)[0][target_label]
        curr_text = tokenizer.decode(input_ids.squeeze(0))

        if debug:
            print(f"iteration: {i}")
            print(f"curr_text: {curr_text}")
            print(f"score: {score}")
    return curr_text


def main(model_name):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    initial_text = utils.utils.random_sentence()
    return gcg(model, tokenizer, initial_text, top_k=256, num_iter=40, target_label=2, debug=True)


if __name__ == '__main__':
    main("cardiffnlp/twitter-roberta-base-sentiment-latest")
