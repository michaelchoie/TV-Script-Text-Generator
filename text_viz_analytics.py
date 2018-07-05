"""Create visualizations and analytics for text data."""

import json
import matplotlib.pyplot as plt
import numpy as np
import operator
import re
import string
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from os import getcwd
from os.path import abspath, join


def get_data():
    """
    Retrieve data from file system.

    Returns
        data (str): text data
    """
    file_name = 'moes_tavern_lines.txt'
    path_to_file = abspath(join(getcwd(), 'data', file_name))

    with open(path_to_file, "r") as f:
        data = f.read()

    return data


def identify_characters(data):
    """
    Get names of all characters in the script.

    Using this method is necessary because it is an unsupervised way of
    finding a character's name, as a character name is revealed
    by a new line and colon (this is how TV scripts are formatted)

    Args
        data (str): text data
    Returns
        characters (list): list of character names
    """
    characters = set(re.findall("\n.*?:", data))
    characters = [x[1:] for x in characters]

    return characters


def count_characters(data, characters):
    """
    Return amount of times a character appears in text.

    Args
        data (str): text data
    Returns
        labels (ndarray):
        values (ndarray):
    """
    count = Counter(data.split())
    character_counts = {name[:-1]: freq for (name, freq) in count.most_common()
                        if name in characters}

    labels, values = zip(*character_counts.items())

    labels = np.array(labels)
    values = np.array(values)

    return labels, values


def text_analytics(data):
    """
    Output summary statistics of the text data.

    Args
        data (str): text data
    """
    num_chars = len(data)
    num_words = len(data.split())
    num_lines = len(re.findall("\n", data))

    print("Number of characters in script: {} \
           Number of words in script: {} \
           Number of lines in script: {}".format(num_chars, num_words,
                                                 num_lines))


def visualize_counts(labels, values):
    """
    Visualize the # of times top 5 characters appear in text using a histogram.

    Args
        labels (str): character names
        values (str): frequency count
    """
    bar_width = 0.5
    indices = np.arange(5)

    plt.bar(indices, values[:5], width=bar_width, align="center")
    plt.xticks(indices, labels[:5])
    plt.title("Top 5 Characters by Frequency")
    plt.show()


def remove_stopwords_names(data, characters):
    """
    Remove stopwords and names for input to wordcloud.

    Args
        data (str): tv script data
        names (list): character names
    Returns
        filtered_script (list): script without stopwords or character names
    """
    strip_punc = str.maketrans('', '', string.punctuation)
    data = data.translate(strip_punc).lower()
    characters = [char.translate(strip_punc).lower() for char in characters]
    word_tokens = word_tokenize(data)
    stop_words = stopwords.words('english')
    filtered_script = [w for w in word_tokens
                       if w not in stop_words and
                       w not in characters and
                       len(w) > 2]

    return filtered_script


def convert_to_json(filtered_script):
    """
    Convert dictionary to JSON for input to R wordcloud script.

    Args
        filtered_script (list): list of filtered words in script
    """
    dictionary = Counter(filtered_script)

    # Use operator.itemgetter for efficiency reasons
    sorted_dict = sorted(dictionary.items(), key=operator.itemgetter(1),
                         reverse=True)
    with open("word_counts.json", "w") as outfile:
        json.dump(sorted_dict, outfile)

    print("JSON stored in {}".format(getcwd()))


def main():
    """Print analytics, visualizations, and create JSON file."""
    data = get_data()
    characters = identify_characters(data)
    labels, values = count_characters(data, characters)
    text_analytics(data)
    visualize_counts(labels, values)
    filtered_script = remove_stopwords_names(data, characters)
    convert_to_json(filtered_script)


if __name__ == "__main__":
    main()
