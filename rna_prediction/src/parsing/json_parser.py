import string
import json


def read_data(filepath):
    """
    Read data from the given file which contains RNA sequences
    and their secondary structure in json notation.
    """
    with open(filepath) as f:
        data = json.load(f)
    return data