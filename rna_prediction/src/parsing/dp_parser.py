import string


OPENING_BRACKETS = '([{<' + string.ascii_uppercase
CLOSING_BRACKETS = ')]}>' + string.ascii_lowercase

def parse_rna_db(lines):
    half = len(lines) // 2

    rna = ''.join(lines[:half])
    db = ''.join(lines[half:])

    allowed_characters = '.' + OPENING_BRACKETS + CLOSING_BRACKETS

    if len(rna) == len(db) and all(c in allowed_characters for c in db):
        yield rna, db


def read_data(filepath):
    """
    Read data from the given file which contains RNA sequences
    and their secondary structure in dot bracket notation.
    """
    allowed_characters = '(.)'

    with open(filepath, 'r') as f:
        lines = []

        for line in f:
            if line and line[0] != '#':
            #if line and line[0] in allowed_characters:
                lines.append(line[:-1])
            elif lines:
                for rna, db in parse_rna_db(lines):
                    yield rna, db

                lines = []
