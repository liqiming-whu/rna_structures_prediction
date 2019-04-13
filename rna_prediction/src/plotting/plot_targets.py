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


arguments = get_arguments(directory=str,
                          extension=(str, 'svg'))

directory, extension = arguments

print("Parsing data...")

with open(f'{directory}/targets.json', 'r') as f:
    data = json.load(f)

target_data = np.asarray([parsing.db_to_matrix(db) for rna, db in data])

os.makedirs(f'{directory}/targets/', exist_ok=True)

print("Iterating through sequences...")

for sequence, targets in enumerate(target_data):
    print(f"Sequence {sequence} of {len(target_data)}")

    length = targets.shape[0]
    mgrid = np.mgrid[:length, :length]

    targets = 10 * targets.flatten() ** 2

    fig, ax = plt.subplots()

    ax.set_xlim(0, length)
    ax.set_ylim(0, length)
    ax.xaxis.tick_top()
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')

    plt.scatter(mgrid[1].flatten(), mgrid[0].flatten(), s=targets, c='b')
    plt.grid()

    plt.savefig(f'{directory}/targets/sequence-{sequence:04d}.{extension}')
    plt.close(fig)
