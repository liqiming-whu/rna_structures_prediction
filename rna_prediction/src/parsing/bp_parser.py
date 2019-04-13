import os
import string


OPENING_BRACKETS = '([{<' + string.ascii_uppercase
CLOSING_BRACKETS = ')]}>' + string.ascii_lowercase


def parse_rna_basepairs(lines):
    rna = ''
    basepairs = []

    allowed_characters = '1234567890'

    for line in lines:
        #if line.strip() and line[0] != '#':
        if line.strip() and line[0] in allowed_characters:
            left, base, right = line.split()

            rna += base
            basepairs.append((int(left), int(right)))

    return rna, basepairs


def bp2dp(basepairs):
    dp = ['.' for _ in basepairs]
    stacks = [[] for _ in OPENING_BRACKETS]

    basepairs.sort()

    for left, right in basepairs:
        if left >= right:
            continue

        for i, s in enumerate(stacks):
            if len(s) == 0 or left >= s[-1] or right <= s[-1]:
                s.append(right)
                dp[left - 1] = OPENING_BRACKETS[i]
                dp[right - 1] = CLOSING_BRACKETS[i]
                break

    return ''.join(dp)


def parse_bpseq(filepath):
    with open(filepath, 'r') as f:
        rna, basepairs = parse_rna_basepairs(f)

        dp = bp2dp(basepairs)

        if dp and len(rna) == len(dp):
            yield rna, dp


def read_data(dirpath):
    """
    Read data from the given directory which contains RNA sequences
    and their secondary structure in bpseq notation.
    """

    for subdir, dirs, files in os.walk(dirpath):
        for filename in files:
            filepath = subdir + os.sep + filename

            if filepath.endswith('.bpseq'):
                for rna, db in parse_bpseq(filepath):
                    yield rna, db
