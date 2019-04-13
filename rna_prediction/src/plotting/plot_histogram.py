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


def parse_bool(b):
    return b.lower() in ['true', 'y', 'yes']


arguments = get_arguments(directory=str,
                          epoch_max=int,
                          epoch_step=int,
                          normalize=(parse_bool, False),
                          extension=(str, 'svg'))

directory, epoch_max, epoch_step, normalize, extension = arguments

print("Parsing data...")

with open(f'{directory}/targets.json', 'r') as f:
    data = json.load(f)

targets = np.asarray([parsing.db_to_matrix(db) for rna, db in data])

os.makedirs(f'{directory}/histograms/', exist_ok=True)

print("Iterating through epochs...")

def flatten(arr):
    return np.hstack([np.asarray(x).flatten() for x in arr])

for epoch in range(0, epoch_max, epoch_step):
    print(f"Epoch {epoch} of {epoch_max}")

    predictions = np.load(f'{directory}/predictions/epoch-{epoch:04d}.npy')

    predictions_flattened = flatten(predictions)
    predictions_negative = flatten(predictions - 2 * targets)
    predictions_positive = flatten(predictions + 2 * targets - 2)

    fig, ax = plt.subplots()

    bins = np.linspace(0, 1, 100)

    plt.hist(predictions_flattened, bins=bins, histtype='step',
            color='b', alpha=0.5, label='overall', density=normalize)
    plt.hist(predictions_negative, bins=bins, histtype='step',
            color='r', alpha=0.5, label='negative', density=normalize)
    plt.hist(predictions_positive, bins=bins, histtype='step',
            color='g', alpha=0.5, label='positive', density=normalize)

    plt.legend()

    plt.savefig(f'{directory}/histograms/epoch-{epoch:04d}.{extension}')
    plt.close(fig)
