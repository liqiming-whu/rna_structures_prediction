import string

import numpy as np


OPENING_BRACKETS = '([{<' + string.ascii_uppercase
CLOSING_BRACKETS = ')]}>' + string.ascii_lowercase


def db_to_matrix(db):

    matrix = np.zeros((len(db), len(db)), dtype=np.uint8)

    stacks = {}

    def stack(opening):
        i = OPENING_BRACKETS.index(opening)
        closing = CLOSING_BRACKETS[i]

        if closing not in stacks:
            stacks[closing] = []

        return stacks[closing]

    for i, c in enumerate(db):
        # opening brackets
        if c in OPENING_BRACKETS:
            stack(c).append(i)
        # closing brackets
        elif c in CLOSING_BRACKETS:
            partner = stacks[c].pop()
            matrix[partner, i] = 1
            matrix[i, partner] = 1
        # no connection
        elif c != '.':
            raise ValueError(f"Illegal character '{c}'")

    return matrix


def rna_to_one_hot(rna):
    one_hots = np.zeros((4, len(rna)))
    chars = 'ACGU'

    char_mapping = {
        'A': 'A',
        'C': 'C',
        'G': 'G',
        'U': 'U',
        'M': 'AC',
        'R': 'AG',
        'W': 'AU',
        'S': 'CG',
        'Y': 'CU',
        'K': 'GU',
        'V': 'ACG',
        'H': 'ACU',
        'D': 'AGU',
        'B': 'CGU',
        'N': 'ACGU'
    }

    for i, c in enumerate(rna):
        # Ignore illegal characters with empty default
        related_chars = char_mapping.get(c, [])

        for h in related_chars:
            # with ambigious characters, don't set the channel to 1, but to a fraction
            one_hots[chars.index(h), i] = 1 / len(related_chars)

    return one_hots
