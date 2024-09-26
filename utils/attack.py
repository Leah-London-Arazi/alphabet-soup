import bert_score
import torch
import numpy as np
import tqdm
import math
import os
import gensim.downloader as api
import transformers
import textattack
from textattack.shared.utils import device as ta_device
from textattack.models.wrappers import HuggingFaceModelWrapper
from utils.defaults import DEFAULT_BATCH_SIZE
from utils.utils import random_sentence


def get_model_wrapper(model_name):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return HuggingFaceModelWrapper(model, tokenizer)


def print_perturbed_result(perturbed_result):
    print(f"perturbed text: '{perturbed_result.attacked_text.text}'.")
    print(f"classified as {perturbed_result.output} with score of {perturbed_result.score}.")
    print(f"used {perturbed_result.num_queries} queries.")


def run_attack(attack, input_text=random_sentence(), label=1):
    attack_result = attack.attack(input_text, label)
    perturbed_result = attack_result.perturbed_result
    print_perturbed_result(perturbed_result)


# Reimplement HuggingFaceModelWrapper method for gradient calculation.
def get_grad_wrt_func(model_wrapper, input_ids, label):
    """
    Receives labels as a parameter to allow performing backprop with user
    provided labels for targeted classification.
    """
    t_label = torch.tensor(label, device=ta_device)
    model = model_wrapper.model

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


def get_bert_score(candidates, word_refs, model_type):
    bert_scorer = bert_score.BERTScorer(model_type=model_type, idf=False, device=ta_device)
    return bert_scorer.score(candidates, word_refs)


def get_bert_avg_score(candidates, word_refs, model_type):
    n_candidates = len(candidates)
    scores = torch.zeros(n_candidates, device=ta_device)

    for word in word_refs:
        bert_scores = get_bert_score(candidates=candidates, word_refs=[word]*n_candidates, model_type=model_type)
        scores += bert_scores[2].to(ta_device)

    return scores / len(word_refs)


def get_bert_max_score(candidates, word_refs, model_type):
    n_candidates = len(candidates)
    scores = torch.zeros(n_candidates, device=ta_device)

    for word in word_refs:
        bert_scores = get_bert_score(candidates=candidates, word_refs=[word]*n_candidates, model_type=model_type)
        scores = torch.max(scores, bert_scores[2].to(ta_device))

    return scores


def get_filtered_token_ids__multi_prefix(model, tokenizer, target_class, confidence_threshold, cache_dir, prefixes, debug):
    # filter embeddings based on classification confidence
    all_token_ids = list(range(len(tokenizer)))
    token_ids = torch.tensor(all_token_ids).cpu()

    for prefix in prefixes:
        cache_file_name = (f"model={model.name_or_path.replace("/", "_")}"
                           f"_target_class={target_class}"
                           f"_confidence_threshold={confidence_threshold}"
                           f"_prefix={prefix}.pt")
        cache_file_path = os.path.join(cache_dir, cache_file_name)

        if os.path.exists(cache_file_path):
            token_ids_prefix = torch.load(cache_file_path)

        else:
            token_ids_prefix = (
                get_filtered_token_ids_by_target_class(model=model,
                                                       tokenizer=tokenizer,
                                                       target_class=target_class,
                                                       confidence_threshold=confidence_threshold,
                                                       prefix=prefix))
            torch.save(token_ids_prefix, cache_file_path)

        token_ids = torch.tensor(np.intersect1d(token_ids_prefix.cpu(), token_ids))

    if token_ids.shape[0] == 0:
        raise Exception("Filtered all tokens!")

    if debug:
        print(f"{len(token_ids)} tokens remaining after filtering")
        filtered_out_token_ids = torch.tensor(np.setdiff1d(all_token_ids, token_ids))
        filtered_out_words = tokenizer.batch_decode([filtered_out_token_ids])
        print(f"Filtered the following tokens: {filtered_out_words}")

    return token_ids


def get_filtered_token_ids(tokenizer, batch_size, filter_func):
    """
    filter_func receives tensor of token_ids and returns a tensor of filtered_token_ids.
    """
    n_tokens = len(tokenizer)
    token_ids = torch.tensor(range(n_tokens)).unsqueeze(1).cpu()
    number_of_batches = math.ceil(n_tokens / batch_size)
    filtered_token_ids = []

    for i in tqdm.trange(number_of_batches):
        token_ids_batch = token_ids[i * batch_size: min((i+1)* batch_size, n_tokens)].to(device=ta_device)
        filtered_token_ids_batch = filter_func(token_ids_batch)
        filtered_token_ids += filtered_token_ids_batch.flatten().tolist()

    return torch.tensor(filtered_token_ids, device=ta_device)


def _filter_by_target_class(token_ids_batch, model, tokenizer, target_class, confidence_threshold, prefix):
    sentences_batch = [f"{prefix} {word}" for word in tokenizer.batch_decode(token_ids_batch)]
    sentences_batch_padded = tokenizer(sentences_batch,
                                       add_special_tokens=True,
                                       return_tensors="pt",
                                       padding=True,
                                       truncation=True).to(device=ta_device)
    confidence = torch.nn.functional.softmax(model(**sentences_batch_padded).logits, dim=1)
    confidence_target_class = confidence[:, target_class]
    return token_ids_batch[confidence_target_class < confidence_threshold]


def get_filtered_token_ids_by_target_class(model, tokenizer, target_class, confidence_threshold, batch_size=1024, prefix=""):
    filter_func = lambda token_ids_batch: _filter_by_target_class(token_ids_batch, model, tokenizer, target_class, confidence_threshold, prefix)
    return get_filtered_token_ids(tokenizer, batch_size, filter_func)


def _filter_by_bert_score(token_ids_batch, tokenizer, word_refs, score_threshold, bert_model_type, debug):
    candidates = tokenizer.batch_decode(token_ids_batch)
    scores = get_bert_max_score(candidates, word_refs, bert_model_type)
    remaining_token_ids = token_ids_batch[scores >= score_threshold]
    if debug:
        remaining_words = tokenizer.batch_decode([remaining_token_ids])
        print(f"The following tokens remained: {remaining_words}")

    return remaining_token_ids


def get_filtered_token_ids_by_bert_score(tokenizer, word_refs, score_threshold,
                                         batch_size=DEFAULT_BATCH_SIZE,
                                         bert_model_type="microsoft/deberta-xlarge-mnli",
                                         debug=False):
    filter_func = lambda token_ids_batch: _filter_by_bert_score(token_ids_batch, tokenizer,
                                                                word_refs, score_threshold,
                                                                bert_model_type, debug)
    return get_filtered_token_ids(tokenizer, batch_size, filter_func)


def get_filtered_token_ids_by_glove_score(tokenizer, word_refs, score_threshold, debug=False):
    vocab = tokenizer.vocab
    cands = []

    glove = api.load('glove-wiki-gigaword-50')

    for word,_ in vocab.items():
        try:
            score = max([glove.similarity(word, ref) for ref in word_refs])
        except KeyError:
            score = 0
        if score >= score_threshold:
            cands.append(word)

    if debug:
        print(f"The following {len(cands)} tokens remained: {cands}")

    # add space to each word
    cands += [f" {w}" for w in cands]

    token_ids = tokenizer(cands, add_special_tokens=False,
                                 return_tensors="np",
                                 padding=True,
                                 truncation=True).input_ids.flatten()

    # remove special token ids
    token_ids = np.setdiff1d(np.unique(token_ids),  tokenizer.all_special_ids)
    return torch.tensor(token_ids, device= ta_device)
