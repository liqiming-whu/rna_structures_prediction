#!/usr/bin/env python3

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import os
import sys

import csv
import numpy as np
from scipy import signal

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import get_arguments


arguments = get_arguments(filename=str,
                          filter=(str,'None'),
                          filter_length=(int, 51),
                          extension=(str, 'svg'))

filename, filter, filter_length, extension = arguments

print("Parsing data...")

epochs = []
epoch_stats = []
lengths = []

with open(f'{filename}', 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')

    # Collect available epochs from header
    first_row = next(csvreader)
    for col_content in first_row[2:]:
        epochs.append(int(col_content))
        epoch_stats.append([])

    for row in csvreader:
        lengths.append(int(row[1]))

        for col, col_content in enumerate(row[2:]):
            epoch_stats[col].append(float(col_content))


print("Plotting graphs...")
print("Length of data: ", len(epoch_stats[0]))

fig, ax = plt.subplots()


for i, epoch in enumerate(epochs):
    if filter != 'None':
        lengths_sort, stats_sort = zip(*sorted(zip(lengths, epoch_stats[i])))
        lengths_sort, stats_sort = (list(t) for t in zip(*sorted(zip(lengths_sort, stats_sort))))
        # filter length, 451 for mean, 351 for median and salgov
        # median filter
        #filt_stats = signal.medfilt(stats_sort,filt_len)
        # mean filter
        filt_stats = np.convolve(stats_sort, np.ones((filter_length,)) / filter_length, mode='valid')
        ones = np.ones((filter_length//2))
        filt_stats = np.concatenate([(ones*filt_stats[0]),filt_stats,(ones*filt_stats[-1])])
        # savgol filter
        #filt_stats = signal.savgol_filter(stats_sort, filt_len, 5)
        plt.plot(lengths_sort, filt_stats, c = "r")
    plt.scatter(lengths, epoch_stats[i], s=15)
    #plt.xlim(0, 300)
    plt.ylim(0, 100)
    plt.savefig(f'{filename}-epoch-{epoch}.{extension}', dpi=400)
    plt.close(fig)
    plt.clf()
    plt.cla()
    plt.close()
    plt.gcf().clear()
