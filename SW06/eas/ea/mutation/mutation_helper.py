from dna.DNA import DNA
from dna.dna_blocks import DenseLayer, ConvLayer, PoolingLayer, FlattenLayer, DropoutLayer


class Error(Exception):
    """Base class for exceptions in this module"""
    pass


class MutationError(Error):
    def __init__(self, message):
        self.message = message


# Get a list of references to the dense layers in the architecture
def getDenseLayerList(dna:DNA):
    layer_list = list()
    for layer in dna.architecture:
        if type(layer) is DenseLayer:
            layer_list.append(layer)
    return layer_list


# Get a list of references to the convolution layers in the architecture
def getConvLayerList(dna:DNA):
    layer_list = list()
    for layer in dna.architecture:
        if type(layer) is ConvLayer:
            layer_list.append(layer)
    return layer_list


# Get a list of references to the pooling layers in the architecture
def getPoolLayerList(dna:DNA):
    layer_list = list()
    for layer in dna.architecture:
        if type(layer) is PoolingLayer:
            layer_list.append(layer)
    return layer_list


# Get a list of references to the dropout layers in the architecture
def getDropoutLayerList(dna:DNA):
    layer_list = list()
    for layer in dna.architecture:
        if type(layer) is DropoutLayer:
            layer_list.append(layer)
    return layer_list


# Split architecture at the flattening layer
def getFlatteningLayerIndex(architecture: list):
    for i in range(len(architecture)):
        if type(architecture[i]) is FlattenLayer:
            return i