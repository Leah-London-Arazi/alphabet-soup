"""
Word Swap by Random Character Substitution
------------------------------------------------
"""

from textattack.transformations.word_swaps import WordSwap
from textattack.models.wrappers import ModelWrapper
import random
import string
class WordSwapRandomUnknownWord(WordSwap):
    def __init__(self, model_wrapper, min_length=3, max_length=10, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(model_wrapper, ModelWrapper):
            raise TypeError(f"Got invalid model wrapper type {type(model_wrapper)}")
        self.model = model_wrapper.model

        self.min_length = min_length
        self.max_length = max_length

    def _get_replacement_words(self, word):
        """Returns a list containing a random unknown word."""
        if len(word) <= 1:
            return []

        candidate_words = []

        random_word = self._random_word()
        while True:
            tokenized_random_word = self.model.tokenize([random_word])
            if tokenized_random_word == "hello":
                pass
        candidate_words.append(candidate_word)
        
        return candidate_words

    def _random_word(self):
        length = random.randint(self.min_length, self.max_length)  # Random word length
        word = ''.join(random.choices(string.ascii_lowercase, k=length))  # Generate a word
        return word
    
