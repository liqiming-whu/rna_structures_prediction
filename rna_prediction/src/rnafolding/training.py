import random
import time

import numpy as np

from . import parsing
from .output import Output

class Training:
    """Training and validation of the neural network"""

    def __init__(self, network, args):
        self.network = network
        self.args = args

    def iterate_minibatches(self, data):
        # Iterate through all the data
        for (rna, db) in data:
            one_hot_rna = parsing.rna_to_one_hot(rna)
            pair_matrix = parsing.db_to_matrix(db)

            # Use singleton list for batch dimension
            yield np.asarray([one_hot_rna]), np.asarray([pair_matrix])

    def transfer_to_cpu(self,network):
        network.inputs.transfer('cpu')
        network.targets.transfer('cpu')

    def do_training(self, data, epoch):
        if not data:
            return

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()

        # Shuffle training data in each epoch for stochastic gradient descent
        random.shuffle(data)

        for batch in self.iterate_minibatches(data):
            inputs, targets = batch
            #if epoch > 0:
            #    print('transfer to cpu')
            #    self.transfer_to_cpu(self.network)
            train_err += self.network.train_fn(inputs, targets)
            train_batches += 1

        # Then we print the results for this epoch:
        print(f"Epoch {epoch + 1} of {self.args.num_epochs} took "
              f"{time.time() - start_time:.3f}s")
        print(f"  {train_err}")
        print(f"  training loss:        {train_err / train_batches:.6f}")

    def do_validation(self, data):
        if not data:
            return 0, 0, 0, 0, 0

        error = accuracy = precision = recall = batches = 0

        predictions = []

        for batch in self.iterate_minibatches(data):
            inputs, targets = batch
            err, acc, pre, rec, pred = self.network.val_fn(inputs, targets)

            error += err
            accuracy += acc
            precision += pre
            recall += rec
            batches += 1

            predictions.append(pred[0, -1])

        avg_error = error / batches
        avg_accuracy = accuracy / batches * 100
        avg_precision = precision / batches * 100
        avg_recall = recall / batches * 100
        if avg_precision+avg_recall > 0:
            avg_f1score = 2 * avg_precision * avg_recall /(avg_precision+avg_recall)
        else:
            avg_f1score = 0
        print("Final results:")
        print(f"  overall loss:        {avg_error:.6f}")
        print(f"  overall accuracy:    {avg_accuracy:.2f} %")
        print(f"  overall precision:   {avg_precision:.2f} %")
        print(f"  overall recall:      {avg_recall:.2f} %")
        print(f"  overall f1score:      {avg_f1score:.2f} %")

        return avg_error, avg_accuracy, avg_precision, avg_recall, avg_f1score, predictions

    def train_network(self, training_data, validation_data):
        # Initialize statistics object
        output = Output(self.args)

        # Save some data
        output.save_objects(args=vars(self.args), targets=validation_data)

        # iterate through epochs
        for epoch in range(self.args.num_epochs):
            # Improve the network with the training data
            self.do_training(training_data, epoch)

            # And perform a full pass over the validation data
            epoch_output = self.do_validation(validation_data)

            # Get current weights from network
            weights = self.network.get_weights()

            # Write everything to the output
            output.save_epoch_output(epoch, epoch_output, weights)
