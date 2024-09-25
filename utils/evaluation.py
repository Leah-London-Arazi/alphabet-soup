import torch


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
