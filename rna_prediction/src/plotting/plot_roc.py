import matplotlib.pyplot as plt
plt.switch_backend('agg')

import json
import os
import sys

from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import get_arguments
from rnafolding import parsing
from statistics import get_TPR_FPR

arguments = get_arguments(directory=str,
                          sequence=int,
                          epoch_max=int,
                          epoch_step=int,
                          extension=(str, 'svg'))

directory, sequence, epoch_max, epoch_step, extension = arguments

os.makedirs(f'{directory}/statistics/', exist_ok=True)

print("Parsing data...")

with open(f'{directory}/targets.json', 'r') as f:
    data = json.load(f)

rna, db = data[sequence]
targets = parsing.db_to_matrix(db)

TPR_list = []
FPR_list = []

print("Iterating through epochs...")

for epoch in range(0, epoch_max, epoch_step):
    print(f"Epoch {epoch} of {epoch_max}")

    prediction_data = np.load(f'{directory}/predictions/epoch-{epoch:04d}.npy')

    '''
    for thresh in np.arange(0.01, 1, thresh_stepsize):
#        t1 = np.array([[0.1, 0.3, 0.1], [0.5, 0.5, 0.6], [0.9, 0.1, 0.1]])
#        t2 = np.array([[0,0,1],[0,1,0],[1,0,0]])
#        TPR,FPR = get_TPR_FPR(t1, t2, thresh)
        TPR, FPR = get_TPR_FPR(prediction_data[sequence], targets, thresh)
        TPR_list.append(TPR)
        FPR_list.append(FPR)

    FPR_sort, TPR_sort = zip(*sorted(zip(FPR_list, TPR_list)))
    FPR_sort, TPR_sort = (list(t) for t in zip(*sorted(zip(FPR_sort, TPR_sort))))
    '''

    tf = targets.flatten()
    pf = prediction_data[sequence].flatten()
    FPR_sort, TPR_sort, thresholds = roc_curve(tf,pf)
    auc = roc_auc_score(tf, pf)
    print("Plotting graph...")

    fig, ax = plt.subplots()

    ax.scatter(FPR_sort, TPR_sort, 10)
    ax.plot(FPR_sort, TPR_sort, 'r')

    ax.set_xlabel('TP Rate')
    ax.set_ylabel('FP Rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_title('Sequence '+str(sequence)+', AUC: '+str(auc))

    print("Saving figure...")

    plt.savefig(f'{directory}/statistics/roc-epoch{epoch:04d}.{extension}')