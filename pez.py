###########
# Adapted from https://github.com/YuxinWenRick/hard-prompts-made-easy/blob/main/optim_utils.py
###########

import random
import numpy as np
import torch
import copy

from sentence_transformers.util import (semantic_search,
                                        dot_score,
                                        normalize_embeddings)


def nn_project(curr_embeds, embedding_layer):
    with torch.no_grad():
        bsz, seq_len, emb_dim = curr_embeds.shape

        # Using the sentence transformers semantic search which is
        # a dot product exact kNN search between a set of
        # query vectors and a corpus of vectors
        curr_embeds = curr_embeds.reshape((-1, emb_dim))
        curr_embeds = normalize_embeddings(curr_embeds)  # queries

        embedding_matrix = embedding_layer.weight
        embedding_matrix = normalize_embeddings(embedding_matrix)

        hits = semantic_search(curr_embeds, embedding_matrix,
                               query_chunk_size=curr_embeds.shape[0],
                               top_k=1,
                               score_function=dot_score)

        nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=curr_embeds.device)
        nn_indices = nn_indices.reshape((bsz, seq_len))

        projected_embeds = embedding_layer(nn_indices)

    return projected_embeds, nn_indices


def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def decode_ids(input_ids, tokenizer, by_token=False):
    input_ids = input_ids.detach().cpu().numpy()

    texts = []

    if by_token:
        for input_ids_i in input_ids:
            curr_text = []
            for tmp in input_ids_i:
                curr_text.append(tokenizer.decode([tmp]))

            texts.append('|'.join(curr_text))
    else:
        for input_ids_i in input_ids:
            texts.append(tokenizer.decode(input_ids_i))

    return texts


def initialize_prompt(tokenizer, token_embedding, prompt_len, device):
    # randomly optimize prompt embeddings
    prompt_ids = torch.randint(len(tokenizer.encoder), (1, prompt_len)).to(device)
    prompt_embeds = token_embedding(prompt_ids).detach()
    prompt_embeds.requires_grad = True

    # initialize the template
    template_text = "{}"
    padded_template_text = template_text.format(" ".join(["<start_of_text>"] * prompt_len))
    dummy_ids = tokenizer.encode(padded_template_text)

    # -1 for optimized tokens
    dummy_ids = [i if i != 49406 else -1 for i in dummy_ids]
    dummy_ids = [49406] + dummy_ids + [49407]
    dummy_ids += [0] * (77 - len(dummy_ids))
    dummy_ids = torch.tensor([dummy_ids]).to(device)

    # for getting dummy embeds; -1 won't work for token_embedding
    tmp_dummy_ids = copy.deepcopy(dummy_ids)
    tmp_dummy_ids[tmp_dummy_ids == -1] = 0
    dummy_embeds = token_embedding(tmp_dummy_ids).detach()
    dummy_embeds.requires_grad = False

    return prompt_embeds, dummy_embeds, dummy_ids


def optimize_prompt_loop(model, tokenizer, token_embedding, num_iters, lr, weight_decay, prompt_len, device):
    # initialize prompt
    prompt_embeds, dummy_embeds, dummy_ids = initialize_prompt(tokenizer, token_embedding, prompt_len, device)
    p_bs, p_len, p_dim = prompt_embeds.shape

    # get optimizer
    input_optimizer = torch.optim.AdamW([prompt_embeds], lr=lr, weight_decay=weight_decay)

    best_text = ""

    for step in range(num_iters):
        # forward projection
        projected_embeds, nn_indices = nn_project(prompt_embeds, token_embedding)

        # tmp_embeds = copy.deepcopy(prompt_embeds)
        tmp_embeds = prompt_embeds.detach().clone()
        tmp_embeds.data = projected_embeds.data
        tmp_embeds.requires_grad = True

        # padding
        # padded_embeds = copy.deepcopy(dummy_embeds)
        padded_embeds = dummy_embeds.detach().clone()
        padded_embeds[dummy_ids == -1] = tmp_embeds.reshape(-1, p_dim)

        loss, _ = model.forward_text_embedding(padded_embeds, dummy_ids) # TODO: our forward and our loss

        prompt_embeds.grad, = torch.autograd.grad(loss, [tmp_embeds])

        input_optimizer.step()
        input_optimizer.zero_grad()

        if step == num_iters - 1:
            best_text = decode_ids(nn_indices, tokenizer)

    return best_text


def optimize_prompt(model, device):
    return optimize_prompt_loop(model=model,
                                tokenizer=model.tokenizer,
                                token_embedding=model.token_embedding,
                                num_iters=0,
                                lr=0,
                                weight_decay=0,
                                prompt_len=0,
                                device=device)
