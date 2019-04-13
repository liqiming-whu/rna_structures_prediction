#!/usr/bin/env python3

import json
import numpy as np
import os
import output
import parsing
import sys


# Generate contact matrix of predictions in given Vienna output file
def generate_prediction_matrices(vienna_file_name):
    sanitized_dps = []

    with open(vienna_file_name, 'r') as vienna_file:
        for i, line in enumerate(vienna_file, start=1):
            if i % 2 == 0:
                curIndex = -1
                while not line[curIndex] == '(':
                    curIndex = curIndex - 1
                sanitized_dps.append(line[0:curIndex-1])

    contact_matrices = []

    for dp in sanitized_dps:
        contact_matrices.append(parsing.db_to_matrix(dp))

    return contact_matrices

# Generate contact matrix of targets in given json file
def generate_target_matrices(target_file_name):

    with open(target_file_name) as data_file:
        data = json.load(data_file)

    target_contact_matrices = []

    for sequence_folding in data:
        target_contact_matrices.append(parsing.db_to_matrix(sequence_folding[1]))

    return target_contact_matrices


def evaluation_fn(predictions, targets):
    true_positives = np.count_nonzero(np.bitwise_and(predictions, targets))
    predicted_positives = np.sum(predictions)
    expected_positives = np.sum(targets)

    #prevent division by zero
    if predicted_positives == 0:
        predicted_positives = 1

    if expected_positives == 0:
        expected_positives = 1

    accuracy = np.count_nonzero(predictions == targets) / targets.size
    precision = true_positives / predicted_positives
    recall = true_positives / expected_positives

    return accuracy, precision, recall, predictions


# Generate average statistics of the predictions
def generate_stats(predictions, targets):
    accuracy = precision = recall = 0

    for prediction, target in zip(predictions, targets):

        acc, pre, rec, pred = evaluation_fn(prediction, target)

        accuracy += acc
        precision += pre
        recall += rec

    avg_error = 0
    avg_accuracy = accuracy / len(targets) * 100
    avg_precision = precision / len(targets) * 100
    avg_recall = recall / len(targets) * 100

    print("Final results:")
    print(f"  overall accuracy:    {avg_accuracy:.2f} %")
    print(f"  overall precision:   {avg_precision:.2f} %")
    print(f"  overall recall:      {avg_recall:.2f} %")

    return avg_error, avg_accuracy, avg_precision, avg_recall, predictions


def save_predictions(predictions, output_path):

        os.makedirs(f'{output_path}/predictions')
        path = f'{output_path}/predictions/epoch-0000'
        np.save(path, np.asarray(predictions))


def main():
    vienna_file_name = sys.argv[1]
    target_file_name = sys.argv[2]
    output_path = sys.argv[3]

    predictions = generate_prediction_matrices(vienna_file_name)
    targets = generate_target_matrices(target_file_name)

    save_predictions(predictions, output_path)


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()


