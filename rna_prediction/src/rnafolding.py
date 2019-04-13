#!/usr/bin/env python3

import argparse
import json

import numpy as np


def check_arguments(args):
    print("You have chosen the following parameters:")

    for arg in sorted(vars(args)):
        print(f'  {arg} = {getattr(args, arg)}')

    if args.output_path is None:
        print("\n"
              "Caution: No output path has been specified, so all output\n"
              "         will be discarded. If you want to save the network\n"
              "         output, please use the --output-path parameter.")


def load_data(filename, maxlen):
    if filename is None:
        return None

    with open(filename, 'r') as f:
        data = [(rna, db) for rna, db in json.load(f) if len(rna) <= maxlen]

    print(f"{filename}: {len(data)} entries read")

    return data


def load_train_val_data(filename, maxlen, val_ratio):
    if filename is None:
        return None

    with open(filename, 'r') as f:
        data = [(rna, db) for rna, db in json.load(f) if len(rna) <= maxlen]

    border = int(len(data) * (1-val_ratio))
    training_data = data[:border]
    if border is not len(data):
        validation_data = data[border+1:]
    else:
        validation_data = []

    # for the case that we want train = val, otherwise COMMENT OUT
    #validation_data = training_data

    print(f"{filename}: {len(data)} entries read")

    return training_data, validation_data


def set_weights(network, weights_file):
    if weights_file is None:
        return

    with np.load(weights_file) as f:
        weights = [f['arr_%d' % i] for i in range(len(f.files))]

    network.set_weights(weights)


def main(args):
    print("Train a deep neural network to perform RNA folding.\n")
    check_arguments(args)

    print("\n-- Loading data...\n")
    if args.training_file is not None:
        if args.validation_file == None:
            training_data, validation_data = load_train_val_data(args.training_file, args.maxlen, args.val_ratio)
        else:
            training_data = load_data(args.training_file, args.maxlen)
            validation_data = load_data(args.validation_file, args.maxlen_validation)
    else:
        training_data = None

    print("\n-- Importing Theano...\n")
    import theano

    print("\n-- Building model and compiling functions...\n")
    from rnafolding.network import NetworkDefinition
    definition = NetworkDefinition(args)
    network = definition.compile_network()

    if args.weights_file is not None:
        set_weights(network, args.weights_file)
        validation_data = load_data(args.test_file, args.maxlen_validation)

    print("\n-- Start training / validation ...\n")
    from rnafolding.training import Training
    training = Training(network, args)
    training.train_network(training_data, validation_data)

    print("\n-- Finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Train a deep neural network to perform RNA folding.",
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
            '--training-data', dest='training_file', type=str,
            help="Database file containing training data")
    parser.add_argument(
            '--validation-data', dest='validation_file', type=str,
            help="Database file containing validation data")
    parser.add_argument(
        '--test-data', dest='test_file', type=str,
        help="Database file containing test data")
    parser.add_argument(
           '--depth', dest='depth', type=int, default=5,
           help="Depth of the network, ie. number of convolutional layers\n"
                "(default: %(default)d)")
    parser.add_argument(
            '--init-weights', dest='weights_file', type=str,
            help="File containing pre-defined weights to be initialize\n"
                 "the network with (default: %(default)s)\n\n"
                 "Important: The weights can only be used with the exact\n"
                 "same network as they were obtained in the first place.\n"
                 "Especially, the convolutional and LSTM layer arguments\n"
                 "and the depth of the network must be the same.")
    parser.add_argument(
            '--maxlen', dest='maxlen', type=int, default=450,
            help="Maximum length of sequences that are considered during training\n"
                 "(default: %(default)d)")
    parser.add_argument(
             '--maxlen-validation', dest='maxlen_validation',
             type=int, default=450,
             help="Maximum length of sequences during validation\n"
                  "(default: %(default)d)")
    parser.add_argument(
             '--mode', dest='mode',
             type=str, default='pool',
             help="Architecture mode, i.e. lstm or pool?\n"
                  "(default: %(default)d)")

    output_group = parser.add_argument_group(title="Output parameters")

    output_group.add_argument(
            '--output-path', dest='output_path', type=str,
            help="Path to write the network output to")
    output_group.add_argument(
            '--output-predictions', dest='output_predictions',
            type=int, default=5,
            help="Epochs in which the network predictions should be stored\n"
                 "(default: %(default)d)")
    output_group.add_argument(
            '--output-weights', dest='output_weights',
            type=int, default=1,
            help="Epochs in which the network weights should be stored\n"
                 "(default: %(default)d)")
    output_group.add_argument(
            '--threshold', dest='threshold', type=float, default=0.5,
            help="The threshold to differentiate between positive and negative network outputs.\n"
                 "(default: %(default)f)")


    # Arguments for network training
    training_group = parser.add_argument_group(title="Training parameters")

    training_group.add_argument(
        '--val-ratio', dest='val_ratio',
        type=float, default=0.2,
        help="Validation ratio for the training data (default: %(default)f)")
    training_group.add_argument(
            '--epochs', dest='num_epochs', type=int, default=100,
            help="Number of epochs (default: %(default)d)")
    training_group.add_argument(
            '--learning-rate', dest='learning_rate',
            type=float, default=5e-5,
            help="Learning rate of the network (default: %(default)f)")
    training_group.add_argument(
            '--loss-func', dest='loss_func', type=str, default='binary-crossentropy',
            choices=['binary-crossentropy', 'squared-error',
                     'binary-hinge-loss'],
            help="Loss function to be used in training (default: %(default)s)\n"
                 "- binary-crossentropy : L = −t log(p) − (1 − t) log(1 − p)\n"
                 "- squared-error       : L = (p - t)^2\n"
                 "- binary-hinge-loss   : L_i = max(0, δ − t_i p_i)")
    training_group.add_argument(
            '--update-func', dest='update_func', type=str, default='rmsprop',
            choices=['adam', 'momentum', 'nesterov', 'rmsprop', 'sgd'],
            help="Loss function to be used in training (default: %(default)s)")
    training_group.add_argument(
            '--weight-decrease', dest='weight_decrease', type=float, default=0.5,
            help="Quotient between the weight of an intermediate output\n"
                 "and its successor (default: %(default)f)")


    # Arguments for convolutional layers
    conv_group = parser.add_argument_group(
            title="Convolutional layer arguments")

    conv_group.add_argument(
            '--filters', dest='num_filters', type=int, default=16,
            help="Number of filters for the convolutional layers\n"
                 "(default: %(default)d)")
    conv_group.add_argument(
            '--filter-size', dest='filter_size', type=int, default=11,
            help="Size of each convolution filter (default: %(default)d)")
    conv_group.add_argument(
            '--filter-size-growth', dest='filter_size_growth',
            type=int, default=0,
            help="Increase the filter size in each step by the given amount\n"
                 "(default: %(default)d)")
    conv_group.add_argument(
            '--filters-shared', dest='num_filters_shared', type=int, default=32,
            help="Number of filters for the convolutional layers\n"
                 "when sharing weights (default: %(default)d)")
    conv_group.add_argument(
            '--share-weights-from', dest='share_weights_from',
            type=int, default=1000,
            help="Index of the block conv layers should start\n"
                 "sharing weights from (default: %(default)d)\n\n"
                 "Note: This works only if filter size growth is set to 0.")


    # Arguments for LSTM layers
    lstm_group = parser.add_argument_group(title="LSTM layer arguments")

    lstm_group.add_argument(
            '--use-lstm', dest='use_lstm', type=str, default='always-bypass',
            choices=['never', 'output', 'always', 'always-bypass'],
            help="Define when to use LSTM layers (default: %(default)s)\n\n"
                 "- never : no LSTM layer is used in the whole network\n"
                 "- output : add a final LSTM layer to the network output\n"
                 "- always : an LSTM transformation is performed each step\n"
                 "- always-bypass : like always but the next conv layer also\n"
                 "                  gets the previous conv output as input")
    lstm_group.add_argument(
            '--units', dest='num_units', type=int, default=16,
            help="Number of units in the LSTM layer (default: %(default)d)")


    main(parser.parse_args())
