from dna.DNA import DNA
from dna.dna_blocks import ConvLayer
from ea.mutation.mutation_helper import MutationError, getConvLayerList, getFlatteningLayerIndex
from numpy.random import choice, randint
import config as cfg

# Conv mutations
# - Insert layer
# - Remove layer
# - Alter stride
# - Alter the number of filters
# - Alter the filter size
# - Change padding
# - Alter the activation function
# - Flip if batch normalization is used or not
# - Flip if a dropout layer is used or not
# - Change the dropout rate

# Variable definition
stride_numbers = cfg.conv_stride_numbers
filter_numbers = cfg.conv_filter_numbers
kernel_sizes = cfg.conv_kernel_sizes
paddings = cfg.conv_paddings
activations = cfg.conv_activations
dropout_rates = cfg.conv_dropout_rates


# Insert a conv layer
def conv_insert_layer(dna: DNA):
    # Generate layer details
    nbr_of_filters = int(choice(filter_numbers))
    kernel_size = int(choice(kernel_sizes))
    stride = int(choice(stride_numbers))
    padding = str(choice(paddings))
    # Generate layer
    new_conv_layer = ConvLayer(filternbr=nbr_of_filters, kernelsize=kernel_size, stride=stride, padding=padding,
                               usebias=True, activation='relu', useBN=False, useDropOut=False, dropoutrate=0.5)
    # Get index of the flattening layer
    flat_index = getFlatteningLayerIndex(architecture=dna.architecture)
    # Generate an index where to insert the new layer into existing architecture
    # This must be between the beginning and the flattening layer
    index_to_insert = randint(flat_index+1)
    # Insert layer into existing architecture
    dna.architecture.insert(index_to_insert, new_conv_layer)


# Remove a conv layer
def conv_remove_layer(dna: DNA):
    # Get a list of the conv layers
    layer_list = getConvLayerList(dna=dna)
    # Check length of the list
    if len(layer_list) == 0:
        # There is no Conv layer left so we throw an error
        raise MutationError("No more conv layer to remove")
    # else: everything ok
    # Randomly select a layer
    layer_to_remove = choice(layer_list)
    # Remove the selected layer from the architecture
    dna.architecture.remove(layer_to_remove)


# Alter the stride of a conv layer
def conv_alter_stride(dna: DNA):
    # Get a list of the conv layers
    layer_list = getConvLayerList(dna=dna)
    # Check length of the list
    if len(layer_list) == 0:
        # There is no Conv layer left so we throw an error
        raise MutationError("No more conv layer to alter")
    # else: everything ok
    # Randomly select a layer
    layer_to_alter = choice(layer_list)
    # Create a list of possible new strides
    possible_strides = [x for x in stride_numbers if x != layer_to_alter.stride]
    # Randomly select a new stride
    new_stride = int(choice(possible_strides))
    # Set the new stride in the selected layer
    layer_to_alter.stride = new_stride


# Alter the number of filters of a conv layer
def conv_alter_filter_nbr(dna: DNA):
    # Get a list of the conv layers
    layer_list = getConvLayerList(dna=dna)
    # Check length of the list
    if len(layer_list) == 0:
        # There is no Conv layer left so we throw an error
        raise MutationError("No more conv layer to alter")
    # else: everything ok
    # Randomly select a layer
    layer_to_alter = choice(layer_list)
    # Create a list of possible new filter numbers
    possible_filter_numbers = [x for x in filter_numbers if x != layer_to_alter.nbr_of_filters]
    # Randomly select a new number of filters
    new_filter_nbr = int(choice(possible_filter_numbers))
    # Set the new number of filters in the selected layer
    layer_to_alter.nbr_of_filters = new_filter_nbr


# Alter the filter size of a conv layer
def conv_alter_filter_size(dna: DNA):
    # Get a list of the conv layers
    layer_list = getConvLayerList(dna=dna)
    # Check length of the list
    if len(layer_list) == 0:
        # There is no Conv layer left so we throw an error
        raise MutationError("No more conv layer to alter")
    # else: everything ok
    # Randomly select a layer
    layer_to_alter = choice(layer_list)
    # Create a list of possible filter sizes
    possible_filter_sizes = [x for x in kernel_sizes if x != layer_to_alter.kernel_size]
    # Randomly select a new filter size
    new_filter_size = int(choice(possible_filter_sizes))
    # Set the new filter size in the selected layer
    layer_to_alter.kernel_size = new_filter_size


# Change the padding of a conv layer
def conv_change_padding(dna: DNA):
    # Get a list of the conv layers
    layer_list = getConvLayerList(dna=dna)
    # Check length of the list
    if len(layer_list) == 0:
        # There is no Conv layer left so we throw an error
        raise MutationError("No more conv layer to alter")
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


# Alter the stride of a conv layer
def conv_alter_activation_function(dna: DNA):
    # Get a list of the conv layers
    layer_list = getConvLayerList(dna=dna)
    # Check length of the list
    if len(layer_list) == 0:
        # There is no Conv layer left so we throw an error
        raise MutationError("No more conv layer to alter")
    # else: everything ok
    # Randomly select a layer
    layer_to_alter = choice(layer_list)
    # Create a list of possible new activations
    possible_activations = activations
    possible_activations.remove(layer_to_alter.activation_type)
    # Randomly select a new activation
    new_activation = str(choice(possible_activations))
    # Set new activation
    layer_to_alter.activation_type = new_activation


# Flip if batch normalization is used or not
def conv_flip_bn_use(dna: DNA):
    # Get a list of the conv layers
    layer_list = getConvLayerList(dna=dna)
    # Check length of the list
    if len(layer_list) == 0:
        # There is no Conv layer left so we throw an error
        raise MutationError("No more conv layer to alter")
    # else: everything ok
    # Randomly select a layer
    layer_to_alter = choice(layer_list)
    # Flip the use of batch normalization in this layer
    layer_to_alter.hasBN = not layer_to_alter.hasBN


# Flip if a dropout layer is used or not
def conv_flip_dropout_use(dna: DNA):
    # Get a list of the conv layers
    layer_list = getConvLayerList(dna=dna)
    # Check length of the list
    if len(layer_list) == 0:
        # There is no Conv layer left so we throw an error
        raise MutationError("No more conv layer to alter")
    # else: everything ok
    # Randomly select a layer
    layer_to_alter = choice(layer_list)
    # Flip the use of dropout in this layer
    layer_to_alter.hasDropOut = not layer_to_alter.hasDropOut


# Change the dropout rate
def conv_change_dropout_rate(dna: DNA):
    # Get a list of the conv layers
    layer_list = getConvLayerList(dna=dna)
    # Check length of the list
    if len(layer_list) == 0:
        # There is no Conv layer left so we throw an error
        raise MutationError("No more conv layer to alter")
    # else: everything ok
    # Randomly select a layer
    layer_to_alter = choice(layer_list)
    # Check if the layer actually uses a dropout function
    if layer_to_alter.hasDropOut:
        # Create a list of possible new dropout rates
        possible_dropouts = [x for x in dropout_rates if x != layer_to_alter.dropout_rate]
        # Randomly select a new dropout rate
        new_dropout_rate = float(choice(possible_dropouts))
        # Change the dropout rate
        layer_to_alter.dropout_rate = new_dropout_rate
    else:
        # Selected layer doesn't use dropout, so we cancel the mutation
        raise MutationError("Dropout in conv layer not activated, therefore rate not changed")
