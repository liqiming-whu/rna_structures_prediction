import json
import os

import numpy as np


class Output:
    """Store relevant data to files for further analysis"""

    def __init__(self, args):
        self.args = args

        self.error = []
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.f1score = []

        if self.args.output_path is not None:
            os.makedirs(f'{self.args.output_path}/predictions')
            os.makedirs(f'{self.args.output_path}/weights')

    def save_objects(self, **kwargs):
        if self.args.output_path is None:
            return

        for name, obj in kwargs.items():
            path = f'{self.args.output_path}/{name}.json'

            with open(path, 'w') as f:
                json.dump(obj, f, indent=2)

    def save_epoch_output(self, epoch, output, weights):
        if self.args.output_path is None:
            return

        error, accuracy, precision, recall, f1score, predictions = output

        self.error.append(error)
        self.accuracy.append(accuracy)
        self.precision.append(precision)
        self.recall.append(recall)
        self.f1score.append(f1score)

        self._save_statistics()

        self._save_predictions(epoch, predictions)
        self._save_weights(epoch, weights)

    def _save_statistics(self):
        path = f'{self.args.output_path}/stats'
        statistics = np.asarray(
            [self.error, self.accuracy, self.precision, self.recall, self.f1score])

        np.save(path, statistics)

    def _matching_epoch(self, epoch, stepsize):
        return stepsize > 0 and epoch % stepsize == 0

    def _save_predictions(self, epoch, predictions):
        if self._matching_epoch(epoch, self.args.output_predictions):
            path = f'{self.args.output_path}/predictions/epoch-{epoch:04d}'
            np.save(path, np.asarray(predictions))

    def _save_weights(self, epoch, weights):
        if self._matching_epoch(epoch, self.args.output_weights):
            path = f'{self.args.output_path}/weights/epoch-{epoch:04d}'
            np.savez(path, *weights)
