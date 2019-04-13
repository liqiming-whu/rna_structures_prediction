#!/usr/bin/env python3

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import get_arguments


arguments = get_arguments(directory=str,
                          sequence=int,
                          epoch_max=int,
                          epoch_step=int,
                          extension=(str, 'svg'))

directory, sequence, epoch_max, epoch_step, extension = arguments

seq_dir = f'sequence-{sequence:04d}'

os.makedirs(f'{directory}/{seq_dir}/', exist_ok=True)

print("Iterating through epochs...")

for epoch in range(0, epoch_max, epoch_step):
    print(f"Epoch {epoch} of {epoch_max}")

    array = np.load(f'{directory}/predictions/epoch-{epoch:04d}.npy')
    predictions = array[sequence]

    length = predictions.shape[0]
    mgrid = np.mgrid[:length, :length]

    predictions = 10 * predictions.flatten() ** 2

    fig, ax = plt.subplots()

    ax.set_xlim(0, length)
    ax.set_ylim(0, length)
    ax.xaxis.tick_top()
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')

    plt.scatter(mgrid[1].flatten(), mgrid[0].flatten(), s=predictions, c='b')
    plt.grid()

    plt.savefig(f'{directory}/{seq_dir}/epoch-{epoch:04d}.{extension}')
    plt.close(fig)
