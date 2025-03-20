from dna.DNA import DNA
from dna.dna_blocks import PoolingLayer
from ea.mutation.mutation_helper import MutationError, getPoolLayerList, getFlatteningLayerIndex
from numpy.random import choice, randint
import config as cfg

# Pooling mutations
# - Insert layer
# - Remove layer
# - Alter the stride
# - Alter the kernel size
# - Change the padding

# Variable definition
stride_numbers = cfg.pool_stride_numbers
kernel_sizes = cfg.pool_kernel_sizes
paddings = cfg.pool_paddings


# Insert a pooling layer
def pool_insert_layer(dna: DNA):
    # Generate layer details
    kernel_size = int(choice(kernel_sizes))
    stride = int(choice(stride_numbers))
    padding = str(choice(paddings))
    # Generate layer
    new_pool_layer = PoolingLayer(poolsize=kernel_size, stride=stride, padding=padding)
    # Get index of the flattening layer
    flat_index = getFlatteningLayerIndex(architecture=dna.architecture)
    # Generate an index where to insert the new layer into existing architecture
    # This must be between the beginning and the flattening layer
    index_to_insert = randint(flat_index + 1)
    # Insert layer into existing architecture
    dna.architecture.insert(index_to_insert, new_pool_layer)


# Remove a pooling layer
def pool_remove_layer(dna: DNA):
    # Get a list of the pooling layers
    layer_list = getPoolLayerList(dna=dna)
    # Check length of the list
    if len(layer_list) == 0:
        # There is no Pooling layer left so we throw an error
        raise MutationError("No more pooling layer to remove")
    # else: everything ok
    # Randomly select a layer
    layer_to_remove = choice(layer_list)
    # Remove the selected layer from the architecture
    dna.architecture.remove(layer_to_remove)


# Alter the stride of a pooling layer
def pool_alter_stride(dna: DNA):
    # Get a list of the pooling layers
    layer_list = getPoolLayerList(dna=dna)
    # Check length of the list
    if len(layer_list) == 0:
        # There is no Pooling layer left so we throw an error
        raise MutationError("No more pooling layer to alter")
    # else: everything ok
    # Randomly select a layer
    layer_to_alter = choice(layer_list)
    # Create a list of possible new strides
    possible_strides = [x for x in stride_numbers if x != layer_to_alter.stride]
    # Randomly select a new stride
    new_stride = int(choice(possible_strides))
    # Set the new stride in the selected layer
    layer_to_alter.stride = new_stride


# Alter the kernel size of a pooling layer
def pool_alter_kernel_size(dna: DNA):
    # Get a list of the pooling layers
    layer_list = getPoolLayerList(dna=dna)
    # Check length of the list
    if len(layer_list) == 0:
        # There is no Pooling layer left so we throw an error
        raise MutationError("No more pooling layer to alter")
    # else: everything ok
    # Randomly select a layer
    layer_to_alter = choice(layer_list)
    # Create a list of possible kernel sizes
    possible_kernel_sizes = [x for x in kernel_sizes if x != layer_to_alter.pool_size]
    # Randomly select a new kernel size
    new_kernel_size = int(choice(possible_kernel_sizes))
    # Set the new filter size in the selected layer
    layer_to_alter.pool_size = new_kernel_size


# Change the padding of a pooling layer
def pool_change_padding(dna: DNA):
    # Get a list of the pooling layers
    layer_list = getPoolLayerList(dna=dna)
    # Check length of the list
    if len(layer_list) == 0:
        # There is no Pooling layer left so we throw an error
        raise MutationError("No more pooling layer to alter")
    # else: everything ok
    # Randomly select a layer
    layer_to_alter = choice(layer_list)
    # Create a list of possible new paddings
    possible_paddings = paddings
    possible_paddings.remove(layer_to_alter.padding)
    # Randomly select a new padding
    new_padding = str(choice(possible_paddings))
    # Set new padding
    layer_to_alter.padding = new_padding
