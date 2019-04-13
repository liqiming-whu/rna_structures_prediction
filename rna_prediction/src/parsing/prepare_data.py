#!/usr/bin/env python3

import json
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import get_arguments

import dp_parser
import bp_parser
import remove_similar_sequences as rss


arguments = get_arguments(path=str,
                          data_format=str,
                          validation_ratio=(float, 0.2),
                          test_ratio=(float, 0.2),
                          path_similar=(str,None))

path, data_format, validation_ratio, test_ratio, path_similar = arguments

# Choose correct parser
if data_format == 'dp':
    read_data = dp_parser.read_data
elif data_format == 'bp':
    read_data = bp_parser.read_data
else:
    print('Invalid format, must be dp or bp.')
    sys.exit()

# Additional parser for the similar dataset
if path_similar is not None:
    if data_format == 'dp':
        read_data = dp_parser.read_data
    elif data_format == 'bp':
        read_data = bp_parser.read_data
    else:
        print('Invalid format, must be dp or bp.')
        sys.exit()

print("Loading the dataset...")

# Load the dataset
data = list(read_data(path))

# remove similar sequences if pathSimilar is set
if path_similar is not None:
    dataSim = list(read_data(path_similar))
    data = rss.remove_sim_seq_list(data,dataSim)

length = len(data)

print(f"  {length} entries read")

# Shuffle the dataset in a random order
# random.shuffle(data)

# Split data into training, validation and test set
validation_length = int(length * validation_ratio)
test_length = int(length * test_ratio)

validation_data = data[:validation_length]
test_data = data[validation_length:validation_length + test_length]
training_data = data[validation_length + test_length:]

print("Some statistics...")

def avg_len(data):
    return sum(len(rna) for rna, db in data) / max(len(data), 1)

print(f"  validation data: {avg_len(validation_data)} // {len(validation_data)}")
print(f"  test data:       {avg_len(test_data)} // {len(test_data)}")
print(f"  training data:   {avg_len(training_data)} // {len(training_data)}")

print("Writing data to files...")

path = path.rstrip('/')

with open(f'{path}-validation.json', 'w') as validation_file:
    json.dump(validation_data, validation_file, indent=2)

with open(f'{path}-test.json', 'w') as test_file:
    json.dump(test_data, test_file, indent=2)

with open(f'{path}-training.json', 'w') as training_file:
    json.dump(training_data, training_file, indent=2)
