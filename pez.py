###########
# Adapted from https://github.com/YuxinWenRick/hard-prompts-made-easy/blob/main/optim_utils.py
###########

import torch
import transformers
from sentence_transformers.util import (semantic_search,
                                        dot_score,
                                        normalize_embeddings)

def nn_project(curr_embeds, token_embeddings, device):
    with torch.no_grad():
        # Using the sentence transformers semantic search which is
        # a dot product exact kNN search between a set of
        # query vectors and a corpus of vectors
        curr_embeds = normalize_embeddings(curr_embeds).squeeze()  # queries

        embedding_matrix = token_embeddings.weight.data
        embedding_matrix = normalize_embeddings(embedding_matrix)

        hits = semantic_search(curr_embeds, embedding_matrix,
                               query_chunk_size=curr_embeds.shape[0],
                               top_k=1,
                               score_function=dot_score)

        nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=curr_embeds.device)

        projected_embeds = token_embeddings(nn_indices).to(device)

    return projected_embeds, nn_indices


def initialize_prompt(token_embedding, prompt_len, device):
    # randomly optimize prompt embeddings
    vocab_size = token_embedding.weight.data.shape[0]
    prompt_ids = torch.randint(vocab_size, (1, prompt_len)).to(device)
    prompt_embeds = token_embedding(prompt_ids).detach()

    return prompt_embeds


def optimize_prompt_loop(model, tokenizer, num_iters, lr, weight_decay, prompt_len, device, debug=False):
    # initialize prompt
    token_embedding = model.get_input_embeddings()
    prompt_embeds = initialize_prompt(token_embedding, prompt_len, device)

    # get optimizer
    input_optimizer = torch.optim.AdamW([prompt_embeds], lr=lr, weight_decay=weight_decay)

    cur_text = ""
    loss = 0
    model.train()
    for step in range(num_iters):
        # forward projection
        projected_embeds, nn_indices = nn_project(prompt_embeds, token_embedding, device)
        projected_embeds.requires_grad = True

        model_output = model(input_ids=nn_indices.unsqueeze(0))
        loss = -1 * model_output.logits.max()
        cur_text = tokenizer.decode(nn_indices)
        prompt_embeds.grad = get_grad(model, tokenizer, cur_text)[:prompt_len].unsqueeze(0)

        if debug:
            print(f"cur_text: {cur_text}")
            print(f"loss: {loss * -1}")

        input_optimizer.step()
        input_optimizer.zero_grad()

    return cur_text, loss * -1

def main(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    best_text = optimize_prompt_loop(model=model,
                         tokenizer=tokenizer,
                         num_iters=20,
                         lr=0.001,
                         weight_decay=1,
                         prompt_len=20,
                         device=device,
                         debug=True)
    print(best_text)

if __name__ == '__main__':
    main("mnoukhov/gpt2-imdb-sentiment-classifier")
