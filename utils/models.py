import torch
import transformers
from textattack.models.helpers import T5ForTextToText
from textattack.shared.utils import device as ta_device
from textattack.models.wrappers import HuggingFaceModelWrapper

from utils.utils import get_logger

logger = get_logger(__name__)


def get_model_wrapper(model_name):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return HuggingFaceModelWrapper(model, tokenizer)


def get_token_ids_without_special_tokens(tokenizer, device="cpu"):
    all_token_ids = list(range(len(tokenizer)))
    special_token_ids = set(tokenizer.all_special_ids)

    # Filter out the special tokens
    token_ids_without_special_tokens = [t for t in all_token_ids if t not in special_token_ids]

    # Convert the list to a tensor and move to the specified device
    token_ids_tensor = torch.tensor(token_ids_without_special_tokens).to(device)

    return token_ids_tensor


def get_words_without_special_tokens(tokenizer):
    special_token_ids = set(tokenizer.all_special_ids)
    return [w for w, t in tokenizer.vocab.items() if t not in special_token_ids]


# Reimplement HuggingFaceModelWrapper method for gradient calculation.
def get_grad_wrt_func(model_wrapper, input_ids, label):
    """
    Receives labels as a parameter to allow performing backprop with user
    provided labels for targeted classification.
    """
    t_label = torch.tensor(label, device=ta_device)
    model = model_wrapper.model

    if isinstance(model, T5ForTextToText):
        raise NotImplementedError(
            "`get_grads` for T5FotTextToText has not been implemented yet."
        )

    model.train()
    embedding_layer = model.get_input_embeddings()
    original_state = embedding_layer.weight.requires_grad
    embedding_layer.weight.requires_grad = True

    emb_grads = []

    def grad_hook(module, grad_in, grad_out):
        emb_grads.append(grad_out[0])

    emb_hook = embedding_layer.register_full_backward_hook(grad_hook)

    model.zero_grad()

    try:
        loss = model(input_ids=input_ids, labels=t_label)[0]
    except TypeError:
        raise TypeError(
            f"{type(model)} class does not take in `labels` to calculate loss. "
            "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
            "(instead of `transformers.AutoModelForSequenceClassification`)."
        )

    loss.backward()

    # grad w.r.t to word embeddings
    grad = emb_grads[0][0].cpu()

    embedding_layer.weight.requires_grad = original_state
    emb_hook.remove()
    model.eval()

    output = {"ids": input_ids, "gradient": grad}

    return output
