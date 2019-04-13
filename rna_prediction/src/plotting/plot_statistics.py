#!/usr/bin/env python3

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import get_arguments


arguments = get_arguments(statistics_file=str,
                          epoch_max=(int, 0),
                          extension=(str, 'svg'))

statistics_file, epoch_max, extension = arguments

print("Loading data...")
statistics = np.load(statistics_file)

epoch_max = epoch_max or statistics.shape[1]

error = statistics[0, :epoch_max]
accuracy = statistics[1, :epoch_max]

if len(statistics) < 5:
    precision = np.zeros_like(error)
    recall = statistics[2, :epoch_max]
    f1score = statistics[3, :epoch_max]
else:
    precision = statistics[2, :epoch_max]
    recall = statistics[3, :epoch_max]
    f1score = statistics[4, :epoch_max]
print("Plotting graphs...")

fig, ax1 = plt.subplots()

plt_error, = ax1.plot(error, 'm--', label='error')
ax1.set_xlabel('epoch')
ax1.set_ylabel('error')

ax2 = ax1.twinx()
plt_accuracy, = ax2.plot(accuracy, 'b-', label='accuracy')
plt_precision, = ax2.plot(precision, 'g-', label='precision')
plt_recall, = ax2.plot(recall, 'r-', label='recall')
plt_f1score, = ax2.plot(f1score, 'c-', label='f1score')
ax2.set_ylabel('percentage')

plt.legend(handles=[plt_error, plt_accuracy, plt_precision, plt_recall, plt_f1score])

print("Saving figure...")

fig.tight_layout()
plt.savefig(f'{statistics_file}.{extension}')
