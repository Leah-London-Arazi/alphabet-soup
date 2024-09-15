import os
import bert_score
import transformers
from sentence_transformers.util import normalize_embeddings
from textattack.models.wrappers import HuggingFaceModelWrapper 
import random
import string
import torch
import numpy as np
import textattack
from textattack.shared.utils import device as ta_device
import tqdm
import math

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
    t_label = torch.tensor(label, device=ta_device)
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
        loss = model(**input_dict, labels=t_label)[0]
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


def get_bert_score(candidates, word_refs):
    bert_scorer = bert_score.BERTScorer(model_type="microsoft/deberta-xlarge-mnli", idf=False, device=ta_device)
    return bert_scorer.score(candidates, word_refs)


def get_bert_avg_score(candidates, word_refs):
    n_candidates = len(candidates)
    scores = torch.zeros(n_candidates, device=ta_device)

    for word in word_refs:
        bert_scores = get_bert_score(candidates=candidates, word_refs=[word]*n_candidates)
        scores += bert_scores[2].to(ta_device)

    return scores / len(word_refs)


def get_filtered_token_ids_single_prefix(model, tokenizer, target_class, confidence_threshold, batch_size, cache_dir, prefix=""):
    cache_file_name = f"model={model.__class__.__name__}_prefix={prefix}.pt"
    cache_file_path = os.path.join(cache_dir, cache_file_name)

    token_embeddings = model.get_input_embeddings()
    token_ids = torch.tensor(range(token_embeddings.num_embeddings), device=ta_device).unsqueeze(1)

    if os.path.exists(cache_file_path):
        confidence = torch.load(cache_file_path)
    else:
        n_tokens = len(token_ids)
        number_of_batches = math.ceil(n_tokens / batch_size)
        confidence = []

        for i in tqdm.trange(number_of_batches):
            token_ids_batch = token_ids[i * batch_size: min((i+1)* batch_size, n_tokens)]
            sentences_batch = [f"{prefix} {word}" for word in tokenizer.batch_decode(token_ids_batch)]
            sentences_batch_padded = tokenizer(sentences_batch,
                                               add_special_tokens=True,
                                               return_tensors="pt",
                                               padding="max_length",
                                               truncation=True).to(device=ta_device)
            confidence_batch = torch.nn.functional.softmax(model(**sentences_batch_padded).logits, dim=1)
            confidence.append(confidence_batch)

        confidence = torch.cat(confidence)

    confidence_target_class = confidence[:, target_class]
    filtered_token_ids = token_ids[confidence_target_class < confidence_threshold].flatten().tolist()

    return torch.tensor(filtered_token_ids, device=ta_device)


def get_filtered_token_ids_multi_prefix(model, tokenizer, target_class, confidence_threshold, cache_dir, prefixes,
                                        batch_size, debug):
    # filter embeddings based on classification confidence
    token_embeddings = model.get_input_embeddings()
    all_token_ids = range(token_embeddings.num_embeddings)
    token_ids = all_token_ids
    for prefix in prefixes:
        token_ids_prefix = get_filtered_token_ids_single_prefix(model=model,
                                                                tokenizer=tokenizer,
                                                                target_class=target_class,
                                                                confidence_threshold=confidence_threshold,
                                                                batch_size=batch_size,
                                                                prefix=prefix,
                                                                cache_dir=cache_dir)
        token_ids = torch.tensor(np.intersect1d(token_ids_prefix, token_ids))

    if token_ids.shape[0] == 0:
        raise Exception("Filtered all tokens!")

    if debug:
        print(f"{len(token_ids)} tokens remaining after filtering")
        filtered_tokens = torch.tensor(np.setdiff1d(all_token_ids, token_ids))
        filtered_words = tokenizer.batch_decode(filtered_tokens)
        print(f"Filtered the following tokens: {filtered_words}")

    filtered_embedding_matrix = normalize_embeddings(token_embeddings(token_ids))
    return token_ids, filtered_embedding_matrix
