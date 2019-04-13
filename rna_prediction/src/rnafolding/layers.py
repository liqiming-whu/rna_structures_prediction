from functools import reduce

import lasagne
import theano
import theano.tensor as T
from lasagne.layers.base import Layer

class GlobalPool1DLayer(Layer):
    """
    GlobalPool1DLayer(incoming,
    pool_function=theano.tensor.mean, **kwargs)
    Global 1-dimensional pooling layer
    This layer pools globally across one dimension, either the third or the forth.
    We assume the follwing dimensions: [minibatch, channels, x, y]
    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.
    pool_function : callable
        the pooling function to use. This defaults to `theano.tensor.mean`
        (i.e. mean-pooling) and can be replaced by any other aggregation
        function.
    axis : integer
        either 2 or 3
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    """

    def __init__(self, incoming, pool_function=T.mean, axis=2, **kwargs):
        super(GlobalPool1DLayer, self).__init__(incoming, **kwargs)
        self.pool_function = pool_function
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        # return input_shape[:2]
        # the dimension of the non-pooled axis describes the output-shape
        if self.axis == 2:
            return input_shape[:2] + (input_shape[3],)
        elif self.axis == 3:
            return input_shape[:2] + (input_shape[2],)

    def get_output_for(self, input, **kwargs):
        if self.axis == 2:
            # input_shape = input.shape()
            # print(input_shape)
            #reshaped = T.swapaxes(input, axis1=2, axis2=3)
            return self.pool_function(input, axis=2)
        elif self.axis == 3:
            return self.pool_function(input, axis=3)

class RepeatConcatLayer(lasagne.layers.MergeLayer):
    """
    Concatenates multiple inputs along the specified axis. Inputs should have
    the same shape except for the dimension specified in axis, which can have
    different sizes. Additionally, if a dimension has size 1 it gets broadcasted
    automatically to the size of the other layers.

    Parameters
    ----------
    incomings : a list of :class:`Layer` instances or tuples
        The layers feeding into this layer, or expected input shapes
    axis : int
        Axis which inputs are joined over
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    """

    def __init__(self, incomings, axis=1, **kwargs):
        super(RepeatConcatLayer, self).__init__(incomings, **kwargs)

        self.axis = axis

    def get_output_shape_for(self, input_shapes):
        # Infer the output shape by grabbing, for each axis, the maximum
        # input size that is not `None` (if there is any)
        output_shape = [None if None in sizes else max(sizes)
                        for sizes in zip(*input_shapes)]

        def match(shape1, shape2):
            axis = self.axis if self.axis >= 0 else len(shape1) + self.axis
            return (len(shape1) == len(shape2) and
                    all(i == axis or s1 is None or s2 is None or s1 == s2 or
                        s1 == 1 or s2 == 1
                        for i, (s1, s2) in enumerate(zip(shape1, shape2))))

        # Check for compatibility with inferred output shape
        if not all(match(shape, output_shape) for shape in input_shapes):
            raise ValueError("Mismatch: input shapes must be the same except "
                             "for the concatenation axis to be broadcasted.")

        # Infer output shape on concatenation axis and return
        sizes = [input_shape[self.axis] for input_shape in input_shapes]
        concat_size = None if any(s is None for s in sizes) else sum(sizes)
        output_shape[self.axis] = concat_size

        return tuple(output_shape)

    def get_output_for(self, inputs, **kwargs):
        # Get output shape
        output_shape = reduce(T.maximum, (input.shape for input in inputs),
                inputs[0].shape)

        def align(input, dim):
            # Check whether we need to repeat this dimension
            should_repeat = T.gt(output_shape[dim], input.shape[dim])
            # Calculate number of repeats
            repeats = T.switch(should_repeat, output_shape[dim], 1)
            # Do not repeat the concatenation axis
            repeats = T.switch(T.neq(dim, self.axis), repeats, 1)
            # Perform the repeat operation
            return T.extra_ops.repeat(input, repeats, axis=dim)

        # Align all dimensions except the concatenation axis
        aligned = [reduce(align, range(input.ndim), input) for input in inputs]

        # Concatenate tensors and return output
        return T.concatenate(aligned, axis=self.axis)

class RepeatSumLayer(lasagne.layers.MergeLayer):
    """
    Sums up multiple inputs along the specified axis. Inputs should have
    the same shape. Additionally, if a dimension has size 1 it gets broadcasted
    automatically to the size of the other layers.

    Parameters
    ----------
    incomings : a list of :class:`Layer` instances or tuples
        The layers feeding into this layer, or expected input shapes
    axis : int
        Axis which inputs are summed up over
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    """

    def __init__(self, incomings, axis=1, **kwargs):
        super(RepeatSumLayer, self).__init__(incomings, **kwargs)

        self.axis = axis

    def get_output_shape_for(self, input_shapes):
        # Infer the output shape by grabbing, for each axis, the maximum
        # input size that is not `None` (if there is any)
        output_shape = [None if None in sizes else max(sizes)
                        for sizes in zip(*input_shapes)]

        def match(shape1, shape2):
            axis = self.axis if self.axis >= 0 else len(shape1) + self.axis
            return (len(shape1) == len(shape2) and
                    all(i == axis or s1 is None or s2 is None or s1 == s2 or
                        s1 == 1 or s2 == 1
                        for i, (s1, s2) in enumerate(zip(shape1, shape2))))

        # Check for compatibility with inferred output shape
        if not all(match(shape, output_shape) for shape in input_shapes):
            raise ValueError("Mismatch: input shapes must be the same except "
                             "for the concatenation axis to be broadcasted.")

        return tuple(output_shape)

    def get_output_for(self, inputs, **kwargs):
        # Get output shape
        output_shape = reduce(T.maximum, (input.shape for input in inputs),
                inputs[0].shape)

        def align(input, dim):
            # Check whether we need to repeat this dimension
            should_repeat = T.gt(output_shape[dim], input.shape[dim])
            # Calculate number of repeats
            repeats = T.switch(should_repeat, output_shape[dim], 1)
            # Do not repeat the concatenation axis
            repeats = T.switch(T.neq(dim, self.axis), repeats, 1)
            # Perform the repeat operation
            return T.extra_ops.repeat(input, repeats, axis=dim)

        # Align all dimensions except the sum axis
        aligned = [reduce(align, range(input.ndim), input) for input in inputs]
        # Sum up tensors and return output
        return T.sum(aligned, axis=self.axis)

def Bidirectional4DLSTMBlock(incoming, axis, **kwargs):
    """
    Performas a bidirectional LSTM transformation on the incoming 4D layer.

    First, the given axis is flattened into the batch dimension.
    Afterwards, two LSTM layer are added which operate in 3D space,
    which walk through the sequence in both directions.
    Finally, the output is reshaped to 4D again by undoing the flattening
    using a reshape layer, assuming the last two dimensions have the same size.

    Parameters
    ----------
    incomings : a :class:`Layer` instance
        The 4D input layer, where the last two dimensions have equal size.
    axis : int
        The axis to iterate over with the LSTM.
    **kwargs
        Any additional keyword arguments are passed to the :class:`LSTMLayer`
        class.

    Notes
    -----
    This layer assumes that the last two dimensions ("spatial dimensions")
    have equal size so it only works with square matrices for each channel.
    """

    # Last value is 2 for axis = 3 and 3 for axis = 2
    pattern = (1, axis, 0, 5 - axis)
    shape = (-1, [2], [1], [1])

    # incoming = (batchsize, channels, length, length)
    dimshuffle = lasagne.layers.DimshuffleLayer(incoming, pattern=pattern)
    # dimshuffle = (channels, length, batchsize, length)
    flatten = lasagne.layers.FlattenLayer(dimshuffle, outdim=3)
    # flatten = (channels, length, length * batchsize)
    dimshuffle = lasagne.layers.DimshuffleLayer(flatten, pattern=(2, 1, 0))
    # dimshuffle =  (length * batchsize, length, channels)
    lstm_forwards = lasagne.layers.LSTMLayer(dimshuffle,
            backwards=False, **kwargs)
    lstm_backwards = lasagne.layers.LSTMLayer(dimshuffle,
            backwards=True, **kwargs)
    # lstm = (batchsize * length, length, units)
    reshape_forwards = lasagne.layers.ReshapeLayer(lstm_forwards, shape=shape)
    reshape_backwards = lasagne.layers.ReshapeLayer(lstm_backwards, shape=shape)
    # reshape = (batchsize, units, length, length)
    return lasagne.layers.ConcatLayer([reshape_forwards, reshape_backwards])
