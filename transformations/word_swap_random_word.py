from textattack.transformations.word_swaps import WordSwap
from utils.utils import random_word

class WordSwapRandomWord(WordSwap):
    def __init__(self, num_random_words=10, rand_word_min_len=3, rand_word_max_length=10, **kwargs):
        super().__init__(**kwargs)
        self.min_length = rand_word_min_len
        self.max_length = rand_word_max_length
        self.num_random_words = num_random_words

    def _get_replacement_words(self, word):
        """
        Returns a list containing a random unknown word.
        """
        if len(word) <= 1:
            return []

        candidate_words = [random_word(self.min_length, self.max_length) for _ in range(self.num_random_words)]

        return candidate_words
