import bert_score
import torch
import numpy as np
import tqdm
import math
import os
import gensim.downloader as api
from textattack.shared.utils import device as ta_device

from consts import FilterTokenIDsMethod
from utils.models import get_token_ids_without_special_tokens, get_words_without_special_tokens
from utils.defaults import BERT_FILTER_DEFAULT_BATCH_SIZE, TARGET_CLASS_FILTER_DEFAULT_BATCH_SIZE, DEFAULT_PREFIXES
from utils.utils import get_logger, get_escaped_model_name

logger = get_logger(__name__)


def get_bert_score(candidates, word_refs, model_type):
    bert_scorer = bert_score.BERTScorer(model_type=model_type, idf=False, device=ta_device)
    return bert_scorer.score(candidates, word_refs)


def get_bert_avg_score(candidates, word_refs, model_type):
    n_candidates = len(candidates)
    scores = torch.zeros(n_candidates, device=ta_device)

    for word in word_refs:
        bert_scores = get_bert_score(candidates=candidates, word_refs=[word] * n_candidates, model_type=model_type)
        scores += bert_scores[2].to(ta_device)

    return scores / len(word_refs)


def get_bert_max_score(candidates, word_refs, model_type):
    n_candidates = len(candidates)
    scores = torch.zeros(n_candidates, device=ta_device)

    for word in word_refs:
        bert_scores = get_bert_score(candidates=candidates, word_refs=[word] * n_candidates, model_type=model_type)
        scores = torch.max(scores, bert_scores[2].to(ta_device))

    return scores


def _filter_by_target_class(token_ids_batch, model, tokenizer, target_class, score_threshold, prefix):
    sentences_batch = [f"{prefix} {word}" for word in tokenizer.batch_decode(token_ids_batch)]
    sentences_batch_padded = tokenizer(sentences_batch,
                                       add_special_tokens=True,
                                       return_tensors="pt",
                                       padding=True,
                                       truncation=True).to(device=ta_device)
    confidence = torch.nn.functional.softmax(model(**sentences_batch_padded).logits, dim=1)
    confidence_target_class = confidence[:, target_class]

    return token_ids_batch[confidence_target_class < score_threshold]


def _filter_by_bert_score(token_ids_batch, tokenizer, word_refs, score_threshold, bert_model_type):
    candidates = tokenizer.batch_decode(token_ids_batch)
    scores = get_bert_max_score(candidates, word_refs, bert_model_type)
    return token_ids_batch[scores >= score_threshold]


def filter_token_ids_by_func(tokenizer, batch_size, filter_func):
    """
    filter_func receives tensor of token_ids and returns a tensor of filtered_token_ids.
    """
    token_ids = get_token_ids_without_special_tokens(tokenizer).unsqueeze(1)
    n_tokens = len(token_ids)
    number_of_batches = math.ceil(n_tokens / batch_size)
    filtered_token_ids = []

    for i in tqdm.trange(number_of_batches):
        token_ids_batch = token_ids[i * batch_size: min((i + 1) * batch_size, n_tokens)].to(device=ta_device)
        filtered_token_ids_batch = filter_func(token_ids_batch)
        filtered_token_ids += filtered_token_ids_batch.flatten().tolist()

    return torch.tensor(filtered_token_ids, device=ta_device)


def get_filtered_token_ids_by_target_class(model, tokenizer, target_class, score_threshold, cache_dir, prefixes):
    # filter embeddings based on classification confidence
    token_ids = get_token_ids_without_special_tokens(tokenizer=tokenizer)
    all_token_ids = token_ids.tolist()

    for prefix in prefixes:
        cache_file_name = (f"model={get_escaped_model_name(model.name_or_path)}"
                           f"_target_class={target_class}"
                           f"_confidence_threshold={score_threshold}"
                           f"_prefix={prefix}.pt")
        cache_file_path = os.path.join(cache_dir, cache_file_name)

        if os.path.exists(cache_file_path):
            logger.debug(f"Used cache file: {cache_file_path}")
            token_ids_prefix = torch.load(cache_file_path)

        else:
            filter_func = lambda token_ids_batch: _filter_by_target_class(token_ids_batch=token_ids_batch,
                                                                          model=model,
                                                                          tokenizer=tokenizer,
                                                                          target_class=target_class,
                                                                          score_threshold=score_threshold,
                                                                          prefix=prefix)
            token_ids_prefix = filter_token_ids_by_func(tokenizer=tokenizer,
                                                        batch_size=TARGET_CLASS_FILTER_DEFAULT_BATCH_SIZE,
                                                        filter_func=filter_func)
            torch.save(token_ids_prefix, cache_file_path)

        token_ids = torch.tensor(np.intersect1d(token_ids_prefix.cpu(), token_ids))

    logger.debug(f"Filtered the following tokens: "
                 f"{tokenizer.batch_decode([torch.tensor(np.setdiff1d(all_token_ids, token_ids))])}")

    return token_ids.to(device=ta_device)


def get_filtered_token_ids_by_bert_score(tokenizer, word_refs, score_threshold,
                                         batch_size=BERT_FILTER_DEFAULT_BATCH_SIZE,
                                         bert_model_type="microsoft/deberta-xlarge-mnli"):
    if len(word_refs) == 0:
        logger.warning("word_refs is empty!!!")
    filter_func = lambda token_ids_batch: _filter_by_bert_score(token_ids_batch=token_ids_batch,
                                                                tokenizer=tokenizer,
                                                                word_refs=word_refs,
                                                                score_threshold=score_threshold,
                                                                bert_model_type=bert_model_type)
    remaining_token_ids = filter_token_ids_by_func(tokenizer, batch_size, filter_func)

    return remaining_token_ids


def get_filtered_token_ids_by_glove_score(tokenizer, word_refs, score_threshold):
    if len(word_refs) == 0:
        logger.warning("word_refs is empty!!!")

    cands = []

    glove = api.load('glove-wiki-gigaword-50')

    for word in get_words_without_special_tokens(tokenizer):
        try:
            score = max([glove.similarity(word, ref) for ref in word_refs])
        except KeyError:
            score = 0
        if score >= score_threshold:
            cands.append(word)

    # add space to each word
    cands += [f" {w}" for w in cands]

    if len(cands) == 0:
        token_ids = []
    else:
        token_ids = tokenizer(cands,
                              add_special_tokens=False,
                              return_tensors="np",
                              padding=True,
                              truncation=True).input_ids.flatten()

    return torch.tensor(token_ids, device=ta_device).unique()


def get_random_tokens(tokenizer, num_tokens, device="cpu"):
    """
    returns random token ids without special tokens.
    """
    token_ids = get_token_ids_without_special_tokens(tokenizer=tokenizer)
    return torch.multinomial(token_ids.float(), num_samples=num_tokens, replacement=False).to(device=device)


def get_filtered_token_ids(filter_method: FilterTokenIDsMethod, model, tokenizer, target_class,
                           cache_dir, word_refs, score_threshold, num_random_tokens=0):
    if filter_method == FilterTokenIDsMethod.by_target_class:
        token_ids = get_filtered_token_ids_by_target_class(model=model,
                                                           tokenizer=tokenizer,
                                                           target_class=target_class,
                                                           score_threshold=score_threshold,
                                                           cache_dir=cache_dir,
                                                           prefixes=DEFAULT_PREFIXES, )

    elif filter_method == FilterTokenIDsMethod.by_bert_score:
        token_ids = get_filtered_token_ids_by_bert_score(tokenizer=tokenizer,
                                                         word_refs=word_refs,
                                                         score_threshold=score_threshold)

    elif filter_method == FilterTokenIDsMethod.by_glove_score:
        token_ids = get_filtered_token_ids_by_glove_score(tokenizer=tokenizer,
                                                          word_refs=word_refs,
                                                          score_threshold=score_threshold)

    elif filter_method == FilterTokenIDsMethod.by_random_tokens:
        token_ids = get_random_tokens(tokenizer, num_random_tokens, device=ta_device)

    else:
        return get_token_ids_without_special_tokens(tokenizer=tokenizer, device=ta_device)

    if token_ids.shape[0] == 0:
        raise RuntimeError("Filtered out all tokens!")

    else:
        logger.debug(f"{len(token_ids)} tokens remaining after filtering")
        logger.debug(f"The following tokens remained: {tokenizer.batch_decode(token_ids.unsqueeze(1), skip_special_tokens=True)}")

    return token_ids

