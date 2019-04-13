#!/usr/bin/env python3

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import get_arguments
from rnafolding import parsing
from statistics import get_accuracy_precision_recall


arguments = get_arguments(directory=str,
                          sequence=int,
                          epoch_max=int,
                          epoch_step=int,
                          threshold=(float, 0.5),
                          extension=(str, 'svg'))

directory, sequence, epoch_max, epoch_step, threshold, extension = arguments

os.makedirs(f'{directory}/statistics/', exist_ok=True)

print("Parsing data...")

with open(f'{directory}/targets.json', 'r') as f:
    data = json.load(f)

rna, db = data[sequence]
targets = parsing.db_to_matrix(db)

accuracy_list = []
precision_list = []
recall_list = []

print("Iterating through epochs...")

for epoch in range(0, epoch_max, epoch_step):
    print(f"Epoch {epoch} of {epoch_max}")

    prediction_data = np.load(f'{directory}/predictions/epoch-{epoch:04d}.npy')
    predictions = prediction_data[sequence]

    data = get_accuracy_precision_recall(predictions, targets, threshold)
    accuracy, precision, recall = data

    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)

print("Plotting graphs...")

fig, ax = plt.subplots()

x = np.arange(0, epoch_max, epoch_step)

plt.plot(x, accuracy_list, 'b-', label='accuracy')
plt.plot(x, precision_list, 'g-', label='precision')
plt.plot(x, recall_list, 'r-', label='recall')

ax.set_xlabel('epoch')
ax.set_ylabel('percentage')
ax.set_ylim(0, 100)

plt.legend()

print("Saving figure...")

plt.savefig(f'{directory}/statistics/sequence-{sequence:04d}.{extension}')
