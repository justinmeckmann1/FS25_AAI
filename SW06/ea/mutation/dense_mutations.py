from dna.DNA import DNA
from dna.dna_blocks import DenseLayer
from ea.mutation.mutation_helper import MutationError, getDenseLayerList, getFlatteningLayerIndex
from numpy.random import choice, randint
import config as cfg

# Dense mutations
# - Insert layer
# - Remove layer
# - Increase the number of neurons
# - Decrease the number of neurons
# - Alter the activation function
# - Flip if a dropout layer is used or not
# - Change the dropout rate

# Variable definition
neuron_numbers = cfg.dense_neuron_numbers
activations = cfg.dense_activations
dropout_rates = cfg.dense_dropout_rates


# Insert a dense layer
def dense_insert_layer(dna: DNA):
    # Generate layer details
    nbr_of_neurons = int(choice(neuron_numbers))
    # Generate layer
    new_dense_layer = DenseLayer(neuronnbr=nbr_of_neurons, usebias=True, activation='relu',
                                 useDropOut=False, dropoutrate=0.5)
    # Get index of the flattening layer
    flat_index = getFlatteningLayerIndex(architecture=dna.architecture)
    # Generate an index where to insert the new layer into existing architecture
    # This must be between the flattening layer and the end
    index_to_insert = randint(flat_index + 1, len(dna.architecture))
    # Insert layer into existing architecture
    dna.architecture.insert(index_to_insert, new_dense_layer)


# Remove a dense layer
def dense_remove_layer(dna: DNA):
    # Get a list of the dense layers
    layer_list = getDenseLayerList(dna=dna)
    # Check length of the list
    if len(layer_list) == 1:
        # There is only one Dense layer left (the last one)
        # so we cannot remove a layer anymore and throw an error
        raise MutationError("No more dense layer to remove")
    # else: everything ok
    # we remove the last layer (softmax), as we cannot alter it
    layer_list.pop()
    # Randomly select a layer
    layer_to_remove = choice(layer_list)
    # Remove the selected layer from the architecture
    dna.architecture.remove(layer_to_remove)


# Increase the number of neurons in a dense layer
def dense_inc_nbr_of_neurons(dna: DNA):
    # Get a list of the dense layers
    layer_list = getDenseLayerList(dna=dna)
    # we remove the last layer (softmax), as we cannot alter it
    layer_list.pop()
    # Check length of the list
    if len(layer_list) == 0:
        # There is no Dense layer left to alter so we throw an error
        raise MutationError("No more dense layer to alter")
    # Randomly select a layer
    layer_to_alter = choice(layer_list)
    # Create a list of possible new neuron numbers
    possible_neuron_nbrs = [x for x in neuron_numbers if x > layer_to_alter.nbr_of_neurons]
    # Randomly select a new nbr of neurons
    new_neuron_nbr = int(choice(possible_neuron_nbrs))
    # Increase the number of neurons in the selected layer
    layer_to_alter.nbr_of_neurons = new_neuron_nbr


# Decrease the number of neurons in a dense layer
def dense_dec_nbr_of_neurons(dna: DNA):
    # Get a list of the dense layers
    layer_list = getDenseLayerList(dna=dna)
    # we remove the last layer (softmax), as we cannot alter it
    layer_list.pop()
    # Check length of the list
    if len(layer_list) == 0:
        # There is no Dense layer left to alter so we throw an error
        raise MutationError("No more dense layer to alter")
    # Randomly select a layer
    layer_to_alter = choice(layer_list)
    # Create a list of possible new neuron numbers
    possible_neuron_nbrs = [x for x in neuron_numbers if x < layer_to_alter.nbr_of_neurons]
    # Randomly select a new nbr of neurons
    new_neuron_nbr = int(choice(possible_neuron_nbrs))
    # Increase the number of neurons in the selected layer
    layer_to_alter.nbr_of_neurons = new_neuron_nbr


# Change the activation function of a dense layer
def dense_alter_activation_function(dna: DNA):
    # Get a list of the dense layers
    layer_list = getDenseLayerList(dna=dna)
    # we remove the last layer (softmax), as we cannot alter it
    layer_list.pop()
    # Check length of the list
    if len(layer_list) == 0:
        # There is no Dense layer left to alter so we throw an error
        raise MutationError("No more dense layer to alter")
    # Randomly select a layer
    layer_to_alter = choice(layer_list)
    # Create a list of possible new activations
    possible_activations = activations
    possible_activations.remove(layer_to_alter.activation_type)
    # Randomly select a new activation
    new_activation = str(choice(possible_activations))
    # Set new activation
    layer_to_alter.activation_type = new_activation


# Flip if a dropout layer is used or not
def dense_flip_dropout_use(dna: DNA):
    # Get a list of the dense layers
    layer_list = getDenseLayerList(dna=dna)
    # we remove the last layer (softmax), as we cannot alter it
    layer_list.pop()
    # Check length of the list
    if len(layer_list) == 0:
        # There is no Dense layer left to alter so we throw an error
        raise MutationError("No more dense layer to alter")
    # Randomly select a layer
    layer_to_alter = choice(layer_list)
    # Flip the use of dropout in this layer
    layer_to_alter.hasDropOut = not layer_to_alter.hasDropOut


# Change the dropout rate
def dense_change_dropout_rate(dna: DNA):
    # Get a list of the dense layers
    layer_list = getDenseLayerList(dna=dna)
    # we remove the last layer (softmax), as we cannot alter it
    layer_list.pop()
    # Check length of the list
    if len(layer_list) == 0:
        # There is no Dense layer left to alter so we throw an error
        raise MutationError("No more dense layer to alter")
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
        raise MutationError("Dropout in dense layer not activated, therefore rate not changed")
