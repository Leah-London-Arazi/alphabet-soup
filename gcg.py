import torch
import transformers
from textattack.shared.utils import device as ta_device
import utils.utils


# Adapted from https://github.com/llm-attacks/llm-attacks
def token_gradients(model, input_ids, t_label):
    embed_weights = model.get_input_embeddings()
    one_hot = torch.zeros(
        input_ids.shape[0],
        embed_weights.weight.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        input_ids.unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    # now stitch it together with the rest of the embeddings
    embeds = embed_weights(input_ids.unsqueeze(0)).detach()
    loss = model(input_ids=input_ids, labels=t_label)[0]

    loss.backward()

    return one_hot.grad.clone()


def sample_control(control_toks, grad, batch_size, topk=256):
    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)
    original_control_toks = control_toks.repeat(batch_size, 1)
    new_token_pos = torch.arange(
        0,
        len(control_toks),
        len(control_toks) / batch_size,
        device=grad.device
    ).type(torch.int64)
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1,
        torch.randint(0, topk, (batch_size, 1),
                      device=grad.device)
    )
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
    return new_control_toks


def gcg_step(model, tokenizer, curr_control, top_k, target_label):
    input_ids = tokenizer(curr_control,
                          add_special_tokens=True,
                          return_tensors="pt",
                          padding=True,
                          truncation=True).input_ids

    grad = token_gradients(model, input_ids, target_label)
    control_encoding = tokenizer(curr_control,
              add_special_tokens=True,
              return_tensors="pt",
              padding=True,
              truncation=True).to(device=ta_device)
    score = torch.nn.functional.softmax(model(**control_encoding).logits, dim=1)[target_label]
    next_control = sample_control(input_ids, grad, 1, top_k)
    # Search
    return next_control, score


def gcg(model, tokenizer, initial_text, top_k, num_iter, target_label, debug=False):
    curr_control = initial_text
    for i in range(num_iter):
        curr_control, score = gcg_step(model, tokenizer, curr_control, top_k, target_label)
        if debug:
            print(f"curr_control: {curr_control}")
            print(f"score: {score}")
    return curr_control


def main(model_name):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    initial_text = utils.utils.random_sentence()
    return gcg(model, tokenizer, initial_text, top_k=256, num_iter=200, target_label=0, debug=False)

if __name__ == '__main__':
    main("cardiffnlp/twitter-roberta-base-sentiment-latest")
