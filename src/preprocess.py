"""Prepare tv script text data so that it can be inputted into a neural net."""

import pickle
from collections import Counter
from os import getcwd
from os.path import abspath, join


class DataPreprocessor(object):
    """
    Preprocess text data for RNN input.

    Contains functions that load data, tokenize punctuation,
    create embedding layer, and handles pickle files

    Attributes:
        file_name (str): name of txt file
        path_to_file (str): path to txt file
    """

    def __init__(self):
        """Initialize object with info on .txt file and path."""
        self.file_name = 'moes_tavern_lines.txt'
        self.path_to_file = abspath(join(getcwd(), '../data',
                                         self.file_name))

    def _load_data(self, path):
        """
        Load data from file.

        Return:
            data (str): text data
        """
        with open(self.path_to_file, "r") as f:
            data = f.read()

        return data

    def _create_lookup_tables(self, text):
        """
        Create lookup tables for vocabulary.

        Args:
            text (str): The text of tv scripts split into words
        Return:
            tuple (vocab_to_int, int_to_vocab)
        """
        word_counts = Counter(text)
        sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
        vocab_to_int = {word: ii for ii, word in enumerate(sorted_words)}
        int_to_vocab = {ii: word for ii, word in enumerate(sorted_words)}
        return (vocab_to_int, int_to_vocab)

    @staticmethod
    def token_lookup():
        """
        Generate a dictionary to turn punctuation into tokens.

        Return:
            token_list (dict): key is symbol, value is token
        """
        token_list = {".": "||Period||",
                      ",": "||Comma||",
                      '"': "|Quotation_Mark||",
                      ";": "||Semicolon||",
                      "!": "||Exclamation_Mark||",
                      "?": "||Question_Mark||",
                      "(": "||Left_Parentheses||",
                      ")": "||Right_Parentheses||",
                      "--": "||Dash||",
                      "\n": "||Return||"
                      }

        return token_list

    def preprocess_and_save_data(self, dataset_path):
        """
        Preprocess and save text data.

        Args:
            dataset_path (str): path to data
        """
        # File starts with copyright line - irrelevant information so remove
        text = self._load_data(dataset_path)
        text = text[81:]

        # Replace punctuation as tokens for text processing
        token_dict = self.token_lookup()
        for key, value in token_dict.items():
            text = text.replace(key, ' {} '.format(value))

        # Transform text for use in lookup tables
        text = text.lower()
        text = text.split()

        # Create lookup tables
        vocab_to_int, int_to_vocab = self._create_lookup_tables(text)
        int_text = [vocab_to_int[word] for word in text]

        pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict),
                    open("preprocess.p", "wb"))

    def load_preprocess(self):
        """Load the preprocessed training data and return it."""
        return pickle.load(open("preprocess.p", "rb"))


def main():
    """Initialize data preprocessor object and save preprocessed data."""
    p = DataPreprocessor()
    p.preprocess_and_save_data(p.path_to_file)


if __name__ == "__main__":
    main()
