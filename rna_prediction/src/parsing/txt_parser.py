def findsize(datafile):
    f = open(datafile, 'r')
    for i, line in enumerate(f):
        pass
    f.close()

    return int((i + 1) / 5)

def getsamples(f, numbers):
    # f: filename
    # number: indices of samples

    numbers = [n * 5 for n in numbers]  # samples take up five lines each
    data = []

    for i, line in enumerate(f):
        if i - 1 in numbers:
            sequence = line.rstrip().split(' ')
            sample = [sequence]
        if i - 2 in numbers:
            structure = line.rstrip().split(' ')
            sample.append(structure)
        if i - 3 in numbers:
            state = line.rstrip().split(' ')
            sample.append(state)
            data.append(sample)

    return data  # returns list of samples, sample is [sequence, structure, state]

file = '/home/marc/PycharmProjects/RNA_Prediction/Data/16Testsequences/rnastateinf-data/data/crw16s-filtered.txt'
size = findsize(file)
f = open(file, 'r')
data = getsamples(f, range(size))
for i in range(size):
    print('4' in data[i][0])
    print('unfinished code')

def read_data(data):
    """
    Read data from the given directory which contains RNA sequences
    and their secondary structure in bpseq notation.
    """
