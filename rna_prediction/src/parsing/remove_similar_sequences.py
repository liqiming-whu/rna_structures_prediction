from time import time
from arguments import get_arguments

arguments = get_arguments(path=str,
                          pathSimilar=(str,None))

path, pathSimilar = arguments

# slide one sequence along another, return largest number of matches
def strmatches(a, b):
    if len(a) > len(b):
        a, b = b, a

    difflist1 = [sum(1 for s, t in zip(a, b[-i:]) if s == t) for i in range(1, len(b) + 1)]
    difflist2 = [sum(1 for s, t in zip(a[i:], b) if s == t) for i in range(1, len(a))]

    return max(difflist1 + difflist2)


# gives longest common substring; modified from
# https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_substring#Python
def longest_common_substring(s1, s2):
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
    return [longest, float(longest) / len(s1), float(longest) / len(s2), x_longest - longest]

# decide whether the two RNA sequences rna and rnaS are too similar
def determine(rna,rnaS):
    l = min(len(rna),len(rnaS))
    d = strmatches(rna, rnaS)
    lcs = longest_common_substring(rna, rnaS)
    if (d > (l * 0.75)) or (lcs[1] > 0.1):
        return True
    else:
        return False



# remove sequences from data that are similar to the ones in dataSim
def remove_sim_seq_list(data,dataSim):
    print(f"Length before: '{len(data)}'")
    i = 1
    for rnaS,dbS in dataSim:
        data[:] = (x for x in data if not determine(x[0],rnaS))
        print(f"FINISHED TEST SEQUENCE: '{i}'")
        i += 1
    print(f"Length afterwards: '{len(data)}'")
    return data
