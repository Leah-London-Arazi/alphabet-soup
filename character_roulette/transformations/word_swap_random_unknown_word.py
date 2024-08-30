"""
Word Swap by Random Character Substitution
------------------------------------------------
"""

from textattack.transformations.word_swaps import WordSwap
import random
import string
class WordSwapRandomUnknownWord(WordSwap):
    def __init__(self, num_random_words=10, min_length=3, max_length=10, **kwargs):
        super().__init__(**kwargs)
        self.min_length = min_length
        self.max_length = max_length
        self.num_random_words = num_random_words

    def _get_replacement_words(self, word):
        """Returns a list containing a random unknown word."""
        if len(word) <= 1:
            return []

        candidate_words = [self._random_word() for _ in range(self.num_random_words)]

        return candidate_words

    def _random_word(self):
        length = random.randint(self.min_length, self.max_length)  # Random word length
        word = ''.join(random.choices(string.ascii_lowercase, k=length))  # Generate a word
        return word

