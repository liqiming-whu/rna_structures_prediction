import collections

import lasagne
import theano
import theano.tensor as T

from . import layers


class Network:

    def __init__(self, network, train_fn, val_fn):
        self.network = network
        self.train_fn = train_fn
        self.val_fn = val_fn

    def get_weights(self):
        return lasagne.layers.get_all_param_values(self.network)

    def set_weights(self, weights):
        lasagne.layers.set_all_param_values(self.network, weights)

class NetworkDefinition:
    """Definition of the RNA folding network."""

    def __init__(self, args):
        self.args = args

        # Define input and target types
        self.inputs = T.tensor3(name='inputs')
        self.targets = T.tensor3(name='targets', dtype='uint8')

        # decide on architecture used
        if self.args.mode == 'pool':
            self.network = self.build_network_pool()
        else:
            self.network = self.build_network()

    def compile_network(self):
        return Network(self.network,
                       self.compile_train_fn(),
                       self.compile_val_fn())

    def get_conv_layer(self, network, prev_conv_layer, block):
        # Increase number of filters when we start to share weights
        if block < self.args.share_weights_from:
            num_filters_block = self.args.num_filters
        else:
            num_filters_block = self.args.num_filters_shared

        # We can only actually share weights two bocks after that
        # because the shared weight matrix expects the increased
        # number of filters as input channels
        if block < self.args.share_weights_from + 2:
            weights = lasagne.init.Normal()
        else:
            weights = prev_conv_layer.W

        if self.args.use_lstm == 'always':
            conv_nonlinearity = lasagne.nonlinearities.leaky_rectify
        else:
            conv_nonlinearity = lasagne.nonlinearities.sigmoid

        filter_size_block = (self.args.filter_size +
                             block * self.args.filter_size_growth)

        # Convolution over the whole sequence matrix
        return lasagne.layers.Conv2DLayer(
            network,
            num_filters=num_filters_block,
            filter_size=filter_size_block,
            stride=1,
            pad='same',
            W=weights,
            nonlinearity=conv_nonlinearity)

    def get_conv_layer_pool(self, network, nonlinearity = True):
        conv_nonlinearity = lasagne.nonlinearities.leaky_rectify

        # Convolution over the whole sequence matrix
        return lasagne.layers.Conv2DLayer(
            network,
            num_filters=self.args.num_filters,
            filter_size=self.args.filter_size,
            stride=1,
            pad='same',
            W=lasagne.init.Normal(),
            nonlinearity=conv_nonlinearity if nonlinearity is True else None)

    def get_lstm_layer(self, network):
        lstm_nonlinearity = lasagne.nonlinearities.sigmoid

        # LSTM expects 3D input with shape (batch_size, length, num_features)
        # rather than 4D
        # Reshape width to batch axis (3D) and after LSTM back to 4D

        # Weight sharing for horizontal and vertical direction
        # No weight sharing for bidirectional LSTM

        # TODO weight sharing
        lstm_horizontal = layers.Bidirectional4DLSTMBlock(network, axis=2,
                num_units=self.args.num_units, nonlinearity=lstm_nonlinearity)
        lstm_vertical = layers.Bidirectional4DLSTMBlock(network, axis=3,
                num_units=self.args.num_units, nonlinearity=lstm_nonlinearity)

        # Pool over the channel/feature dimension and append it to the overall
        # output, then in the loss function, compare each with the target
        # Start in layer i (eg. last five layers) to calculate loss
        return lasagne.layers.ConcatLayer([lstm_horizontal, lstm_vertical])

    def get_pool_and_broadcasting_layer(self, network):
        # split block into horiz/vertic/original blocks
        # apply convolutions, concatenate and apply nonlinearity, see pres slides
        horiz = layers.GlobalPool1DLayer(network, axis=2)
        horiz = lasagne.layers.Conv1DLayer(
            horiz,
            num_filters=self.args.num_filters,
            filter_size=1,
            stride=1,
            pad='same',
            W=lasagne.init.Normal(),
            nonlinearity=None)
        horiz = lasagne.layers.DimshuffleLayer(
            horiz, pattern=(0, 1, 'x', 2))
        vertic = layers.GlobalPool1DLayer(network, axis=3)
        vertic = lasagne.layers.Conv1DLayer(
            vertic,
            num_filters=self.args.num_filters,
            filter_size=1,
            stride=1,
            pad='same',
            W=lasagne.init.Normal(),
            nonlinearity=None)
        vertic = lasagne.layers.DimshuffleLayer(
            vertic, pattern=(0, 1, 2, 'x'))
        network = lasagne.layers.Conv2DLayer(
            network,
            num_filters=self.args.num_filters,
            filter_size=1,
            stride=1,
            pad='same',
            W=lasagne.init.Normal(),
            nonlinearity=None)
        network = layers.RepeatConcatLayer([network, horiz,
                                            vertic])
        network = lasagne.layers.NonlinearityLayer(network, nonlinearity=lasagne.nonlinearities.leaky_rectify)
        feature_pool = lasagne.layers.FeaturePoolLayer(network,
                                                       axis=1, pool_size=network.output_shape[1])
        return network, feature_pool

    def build_network_pool(self):
        # Input layer consisting of the dimensions
        # (batchsize, 4 one-hot encoded channels, length)
        # CHANGE None to 1 for batchsize for optimization
        input_flat = lasagne.layers.InputLayer(
            shape=(None, 4, None), input_var=self.inputs)

        # Repeat the vectors horizontally and vertically
        input_horizontal = lasagne.layers.DimshuffleLayer(
            input_flat, pattern=(0, 1, 'x', 2))
        input_vertical = lasagne.layers.DimshuffleLayer(
            input_flat, pattern=(0, 1, 2, 'x'))

        # Concatenate horizontal and vertical channels
        input_broadcasted = layers.RepeatConcatLayer([input_horizontal,
                                                      input_vertical])

        # Initial convolutional layer which works only on the broadcasted input
        network = input_broadcasted

        # Collect output values from each step
        outputs = []

        # Repeat the convolutional layer
        for block in range(self.args.depth):
            # Get the convolutional layer
            conv_layer = self.get_conv_layer_pool(network)

            # Concatenate the original input with the output of the
            # convolutional layer
            network = lasagne.layers.ConcatLayer([input_broadcasted,
                                                  conv_layer])

            # Add result to the intermediate output list
            feature_pool = lasagne.layers.FeaturePoolLayer(conv_layer,
                                                           axis=1, pool_size=conv_layer.output_shape[1])
            outputs.append(feature_pool)

        for block in range(1):
            # do final pooling and broadcasting
            network, feature_pool = self.get_pool_and_broadcasting_layer(network)
            network = lasagne.layers.ConcatLayer([input_broadcasted,
                                                  network])
            outputs.append(feature_pool)

        output_fin = lasagne.layers.ConcatLayer(outputs)
        # concatenate the list of feature-pool-generated outputs into one tensor
        return output_fin

    def build_network(self):
        # Input layer consisting of the dimensions
        # (batchsize, 4 one-hot encoded channels, length)
        input_flat = lasagne.layers.InputLayer(
                shape=(None, 4, None), input_var=self.inputs)

        # Repeat the vectors horizontally and vertically
        # to provide the cross product over all characters
        input_horizontal = lasagne.layers.DimshuffleLayer(
                input_flat, pattern=(0, 1, 'x', 2))
        input_vertical = lasagne.layers.DimshuffleLayer(
                input_flat, pattern=(0, 1, 2, 'x'))

        # Concatenate horizontal and vertical channels
        input_broadcasted = layers.RepeatConcatLayer([input_horizontal,
                                                      input_vertical])

        # Initial convolutional layer which works only on the broadcasted input
        network = input_broadcasted
        conv_layer = None

        # Collect output values from each step
        outputs = []

        # Repeat the convolutional recurrent block
        for block in range(self.args.depth):
            # Get the convolutional layer
            conv_layer = self.get_conv_layer(network, conv_layer, block)

            # Concatenate the original input with the output of the
            # convolutional network
            network = lasagne.layers.ConcatLayer([input_broadcasted,
                                                  conv_layer])

            if self.args.use_lstm in ['always', 'always-bypass']:
                # Get the LSTM layer
                lstm_layer = self.get_lstm_layer(network)

                # Concatenate the original input with the output of the
                # LSTM layers and the output of the previous conv layer
                if self.args.use_lstm == 'always':
                    network_layers = [input_broadcasted, lstm_layer]
                    feature_layer = lstm_layer
                else:
                    network_layers = [input_broadcasted, lstm_layer, conv_layer]
                    feature_layer = conv_layer

                network = lasagne.layers.ConcatLayer(network_layers)
            else:
                feature_layer = conv_layer

            # Add result to the intermediate output list
            feature_pool = lasagne.layers.FeaturePoolLayer(feature_layer,
                    axis=1, pool_size=feature_layer.output_shape[1])
            outputs.append(feature_pool)

        if self.args.use_lstm == 'output':
            network = lasagne.layers.ConcatLayer([input_broadcasted] + outputs)

            # Add a final LSTM layer for post processing
            lstm_layer = self.get_lstm_layer(network)

            # Add result to the intermediate output list
            feature_pool = lasagne.layers.FeaturePoolLayer(lstm_layer,
                    axis=1, pool_size=feature_layer.output_shape[1])
            outputs.append(feature_pool)

        # concatenate the list of feature-pool-generated outputs into one tensor
        return lasagne.layers.ConcatLayer(outputs)

    # all contained 3d tensors have the same ValueError
    # the last matrix of each 3d tensor has the value '1' in each element
    # each previous matrix is the previous matrix * factor
    def generate_pow3_tensor(self, A):
        # Define scan function
        result, _ = theano.scan(
                fn=lambda _, prior: prior * self.args.weight_decrease,
                sequences=[A[:-1]],
                outputs_info=T.ones_like(A[0]))

        constant_ones_channel = T.ones_like(A[0]).dimshuffle('x', 0, 1)
        pow3_tensor = T.concatenate((constant_ones_channel, result))

        return pow3_tensor[::-1]

    # generates a weight tensor of same dimension as A
    # a weight matrix is a 4D tensor of equal pow3_tensors
    def generate_weights(self, A):
        result, _ = theano.map(fn=self.generate_pow3_tensor, sequences=[A])

        return result

    def get_loss_func(self, predictions, targets):
        # Clip predictions as to prevent nan caused by calculating log(0)
        clipped_predictions = T.clip(predictions, 0.000001, 0.999999)

        if self.args.loss_func == 'binary-crossentropy':
            # Calculate loss based on binary crossentropy
            return lasagne.objectives.binary_crossentropy(
                    clipped_predictions, targets)
        elif self.args.loss_func == 'squared-error':
            # Calculate loss based on squared error
            return lasagne.objectives.squared_error(
                    clipped_predictions, targets)
        elif self.args.loss_func == 'binary-hinge-loss':
            # Calculate loss based on binary hinge loss
            return lasagne.objectives.binary_hinge_loss(
                    clipped_predictions, targets, log_odds=False)

        raise ValueError(f"Unknown loss function '{self.args.loss_func}'")


    def get_loss(self, predictions):
        dimshuffled_targets = self.targets.dimshuffle(0, 'x', 1, 2)

        # Determine basic loss function
        loss = self.get_loss_func(predictions, dimshuffled_targets)

        # Create a weight matrix
        weights = self.generate_weights(predictions)

        # Calculate imbalance in targets
        ones = T.sum(dimshuffled_targets)
        length = dimshuffled_targets.shape[2]
        imbalance = (length * length - ones) / ones
        correction = dimshuffled_targets + 1 / (imbalance - 1)

        # Calculate imbalance based on global function
        # length = targets.shape[2]
        # imbalance = factor * length * length

        # Weight the error and multiply with target value to increase
        # to increase the punishment for missing ones
        weighted_error = loss * weights * correction / length

        # We use sum instead of mean because the samples have different length
        return T.sum(weighted_error)

    def get_binary_predictions(self, predictions):
        binary_predictions = predictions[:, -1] > self.args.threshold

        return T.cast(binary_predictions, dtype='uint8')

    def compile_train_fn(self):
        # Define predictions and loss function
        predictions = lasagne.layers.get_output(self.network)
        loss = self.get_loss(predictions)

        # Get the network parameters and allow to update them
        params = lasagne.layers.get_all_params(self.network, trainable=True)

        # Set the update function based on the hyper parameter
        if self.args.update_func == 'adam':
            updates = lasagne.updates.adam(loss, params,
                    learning_rate=self.args.learning_rate)
        elif self.args.update_func == 'momentum':
            updates = lasagne.updates.momentum(loss, params,
                    learning_rate=self.args.learning_rate)
        elif self.args.update_func == 'nesterov':
            updates = lasagne.updates.nesterov_momentum(loss, params,
                    learning_rate=self.args.learning_rate)
        elif self.args.update_func == 'rmsprop':
            updates = lasagne.updates.rmsprop(loss, params,
                    learning_rate=self.args.learning_rate)
        elif self.args.update_func == 'sgd':
            updates = lasagne.updates.sgd(loss, params,
                    learning_rate=self.args.learning_rate)

        # Create the training function used in the batch training
        return theano.function(
            [self.inputs, self.targets],
            loss,
            updates=updates,
            allow_input_downcast=True)

    def compile_val_fn(self):
        # Create a loss expression for validation/testing.
        predictions = lasagne.layers.get_output(
                self.network, deterministic=True)
        loss = self.get_loss(predictions)

        # Define accuracy and recall
        binary_predictions = self.get_binary_predictions(predictions)
        true_positives = T.bitwise_and(binary_predictions, self.targets).sum()
        predicted_positives = binary_predictions.sum()
        expected_positives = self.targets.sum()

        accuracy = T.eq(binary_predictions, self.targets).mean()
        precision = true_positives / T.clip(predicted_positives, 1, 1e20)
        recall = true_positives / T.clip(expected_positives, 1, 1e20)

        # Create an evaluation function as well
        return theano.function(
            [self.inputs, self.targets],
            [loss, accuracy, precision, recall, predictions],
            allow_input_downcast=True)
