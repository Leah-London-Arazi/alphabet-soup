import transformers
from textattack.models.wrappers import HuggingFaceModelWrapper 
import random
import string
import torch
import numpy as np
import textattack

def get_model_wrapper(model_name):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return HuggingFaceModelWrapper(model, tokenizer)


def print_perturbed_result(perturbed_result):
    print(f"perturbed text: '{perturbed_result.attacked_text.text}'.")
    print(f"classified as {perturbed_result.output} with score of {perturbed_result.score}.")
    print(f"used {perturbed_result.num_queries} queries.")


def random_word(min_len=3, max_len=10):
    length = random.randint(min_len, max_len)  # Random word length
    characters = string.digits + string.ascii_letters + string.punctuation
    word = ''.join(random.choices(characters, k=length))  # Generate a word
    return word


def random_sentence(min_len=3, max_len=10):
    sen_length = random.randint(min_len, max_len)  # Random sentence length
    return " ".join([random_word() for _ in range(sen_length)])


def run_attack(attack, input_text=random_sentence(), label=1):
    attack_result = attack.attack(input_text, label)
    perturbed_result = attack_result.perturbed_result
    print_perturbed_result(perturbed_result)


def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


# Reimplement HuggingFaceModelWrapper method for gradient calculation.
# Get labels as a parameter to allow performing backprop with user provided labels for targeted classification
def get_grad_wrt_func(model_wrapper, text_input, label):
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer

    if isinstance(model, textattack.models.helpers.T5ForTextToText):
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

    emb_hook = embedding_layer.register_backward_hook(grad_hook)

    model.zero_grad()
    model_device = next(model.parameters()).device
    input_dict = tokenizer(
        [text_input],
        add_special_tokens=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
    input_dict.to(model_device)

    try:
        loss = model(**input_dict, labels=label)[0]
    except TypeError:
        raise TypeError(
            f"{type(model)} class does not take in `labels` to calculate loss. "
            "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
            "(instead of `transformers.AutoModelForSequenceClassification`)."
        )

    loss.backward()

    # grad w.r.t to word embeddings
    grad = emb_grads[0][0].cpu().numpy()

    embedding_layer.weight.requires_grad = original_state
    emb_hook.remove()
    model.eval()

    output = {"ids": input_dict["input_ids"], "gradient": grad}

    return output


def char_level_entropy(text):
    # Create a set of unique characters
    unique_chars = list(set(text))

    # Convert characters to indices
    indices = torch.tensor(list(range(len(unique_chars))), dtype=torch.long)

    # Count character occurrences
    num_chars = len(text)
    counts = torch.bincount(indices, minlength=len(unique_chars))

    # Convert counts to probabilities
    probabilities = counts.float() / num_chars

    return torch.sum(torch.special.entr(probabilities))
