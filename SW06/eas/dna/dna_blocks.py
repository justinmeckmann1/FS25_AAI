class ConvLayer(object):
    def __init__(self, filternbr=1, kernelsize=3, stride=1, padding='valid', usebias=True, activation='linear',
                 useBN=True, useDropOut=False, dropoutrate=0.5):
        self.nbr_of_filters = filternbr
        self.kernel_size = kernelsize
        self.stride = stride
        self.padding = padding
        self.use_bias = usebias
        self.activation_type = activation
        self.hasBN = useBN
        self.hasDropOut = useDropOut
        self.dropout_rate = dropoutrate


class DenseLayer(object):
    def __init__(self, neuronnbr=1, usebias=True, activation='linear', useDropOut=False, dropoutrate=0.5):
        self.nbr_of_neurons = neuronnbr
        self.use_bias = usebias
        self.activation_type = activation
        self.hasDropOut = useDropOut
        self.dropout_rate = dropoutrate


class PoolingLayer(object):
    def __init__(self, poolsize=2, stride=2, padding='valid'):
        self.pool_size = poolsize
        self.stride = stride
        self.padding = padding


class FlattenLayer(object):
    def __init__(self):
        pass


class DropoutLayer(object):
    def __init__(self, rate=0.5):
        self.dropout_rate = rate


class BatchNormLayer(object):
    def __init__(self):
        pass
