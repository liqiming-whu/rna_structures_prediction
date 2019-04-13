#!/usr/bin/env python3

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from arguments import get_arguments

from parsing import dp_parser
from parsing import bp_parser
from parsing import json_parser
from rnafolding import parsing




arguments = get_arguments(filename=str,
                          data_format=str,
                          minlen=(int, 0),
                          maxlen=(int, 100000),
                          extension=(str, 'svg'))

filename, data_format, minlen, maxlen, extension = arguments

# Choose correct parser
if data_format == 'dp':
    read_data = dp_parser.read_data
elif data_format == 'bp':
    read_data = bp_parser.read_data
elif data_format == 'json':
    read_data = json_parser.read_data
else:
    print('Invalid format, must be dp or bp.')
    sys.exit()


print("Parsing data...")

length_list = []
contacts_list = []
imbalance_list = []

if(data_format == 'json'):
    data = read_data(filename)
    for item in data:
        rna = item[0]
        db = item[1]
        length = len(rna)
        # print(length)

        if length < minlen or maxlen < length:
            continue

        contacts = sum(1 for x in db if x in parsing.OPENING_BRACKETS)
        imbalance = 2 * contacts / (length * length)

        length_list.append(length)
        contacts_list.append(contacts)
        imbalance_list.append(imbalance)
else:
    for rna, db in read_data(filename):
        length = len(rna)
       # print(length)

        if length < minlen or maxlen < length:
            continue

        contacts = sum(1 for x in db if x in parsing.OPENING_BRACKETS)
        imbalance = 2 * contacts / (length * length)

        length_list.append(length)
        contacts_list.append(contacts)
        imbalance_list.append(imbalance)

print("Length of data: "+str(len(length_list)))
print("Mean length: "+str(np.mean(length_list)))
print("Median length: "+str(np.median(length_list)))

print("Plotting graphs...")

fig, ax = plt.subplots()

plt.scatter(length_list, contacts_list, s=0.4)
plt.savefig(f'{filename}-contacts-per-length.{extension}')
plt.close(fig)


fig, ax = plt.subplots()

plt.scatter(length_list, imbalance_list, s=0.4)
plt.savefig(f'{filename}-imbalance-per-length.{extension}')
plt.close(fig)
