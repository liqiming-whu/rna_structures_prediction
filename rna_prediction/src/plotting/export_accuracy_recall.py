#!/usr/bin/env python3

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import csv
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import get_arguments
from rnafolding import parsing
from statistics import get_accuracy_precision_recall


arguments = get_arguments(directory=str,
                          epoch_max=int,
                          epoch_step=int,
                          threshold=(float, 0.5))

directory, epoch_max, epoch_step, threshold = arguments

print("Parsing data...")

with open(f'{directory}/targets.json', 'r') as f:
    data = json.load(f)

target_data = np.asarray([parsing.db_to_matrix(db) for rna, db in data])

print("Iterating through epochs...")


n = len(target_data)

accuracy_list = [[] for _ in range(n)]
precision_list = [[] for _ in range(n)]
recall_list = [[] for _ in range(n)]


for epoch in range(0, epoch_max, epoch_step):
    print(f"Epoch {epoch} of {epoch_max}")

    prediction_data = np.load(f'{directory}/predictions/epoch-{epoch:04d}.npy')

    for i, predictions, targets in zip(range(n), prediction_data, target_data):
        data = get_accuracy_precision_recall(predictions, targets, threshold)
        accuracy, precision, recall = data

        accuracy_list[i].append(accuracy)
        precision_list[i].append(precision)
        recall_list[i].append(recall)


with open(f'{directory}/accuracy.csv', 'w') as accuracy_file, \
        open(f'{directory}/precision.csv', 'w') as precision_file, \
        open(f'{directory}/recall.csv', 'w') as recall_file:
    accuracy_writer = csv.writer(accuracy_file)
    precision_writer = csv.writer(precision_file)
    recall_writer = csv.writer(recall_file)

    epochs = list(range(0, epoch_max, epoch_step))

    accuracy_writer.writerow(['#', 'length'] + epochs)
    precision_writer.writerow(['#', 'length'] + epochs)
    recall_writer.writerow(['#', 'length'] + epochs)

    data = zip(range(n), accuracy_list, precision_list, recall_list)

    for i, accuracy, precision, recall in data:
        length = len(target_data[i])
        accuracy_writer.writerow([i, length] + accuracy)
        precision_writer.writerow([i, length] + precision)
        recall_writer.writerow([i, length] + recall)
