from dna.DNA import DNA
from dna.dna_blocks import DropoutLayer
from ea.mutation.mutation_helper import MutationError, getDropoutLayerList
from numpy.random import choice, randint
import config as cfg

# Dropout mutations
# - Insert layer
# - Remove layer
# - Alter the dropout rate

# Variable definition
rates = cfg.drop_dropout_rates


# Insert a dropout layer
def dropout_add_layer(dna: DNA):
    # Generate layer details
    rate = float(choice(rates))
    # Generate layer
    new_dropout_layer = DropoutLayer(rate=rate)
    # Get the number of layers in the network
    nbr_of_layers = len(dna.architecture)
    # Generate an index where to insert the new layer into existing architecture
    # This must be between the beginning and one before the end layer
    index_to_insert = randint(nbr_of_layers - 1)
    # Insert layer into existing architecture
    dna.architecture.insert(index_to_insert, new_dropout_layer)


# Remove a dropout layer
def dropout_remove_layer(dna: DNA):
    # Get a list of the dropout layers
    layer_list = getDropoutLayerList(dna=dna)
    # Check length of the list
    if len(layer_list) == 0:
        # There is no Dropout layer left so we throw an error
        raise MutationError("No more dropout layer to remove")
    # else: everything ok
    # Randomly select a layer
    layer_to_remove = choice(layer_list)
    # Remove the selected layer from the architecture
    dna.architecture.remove(layer_to_remove)


# Alter the rate of a dropout layer
def dropout_alter_rate(dna: DNA):
    # Get a list of the dropout layers
    layer_list = getDropoutLayerList(dna=dna)
    # Check length of the list
    if len(layer_list) == 0:
        # There is no Dropout layer left so we throw an error
        raise MutationError("No more dropout layer to alter")
    # else: everything ok
    # Randomly select a layer
    layer_to_alter = choice(layer_list)
    # Create a list of possible new rates
    possible_rates = [x for x in rates if x != layer_to_alter.dropout_rate]
    # Randomly select a new stride
    new_rate = float(choice(possible_rates))
    # Set the new rate in the selected layer
    layer_to_alter.dropout_rate = new_rate
