from dna.DNA import DNA
from numpy.random import choice
from ea.mutation.mutation_helper import MutationError
from ea.mutation.dense_mutations import dense_insert_layer, dense_remove_layer, dense_inc_nbr_of_neurons, \
    dense_dec_nbr_of_neurons, dense_alter_activation_function, dense_flip_dropout_use, dense_change_dropout_rate
from ea.mutation.conv_mutations import conv_insert_layer, conv_remove_layer, conv_alter_stride, conv_alter_filter_nbr, \
    conv_alter_filter_size, conv_change_padding, conv_alter_activation_function, conv_flip_bn_use, \
    conv_flip_dropout_use, conv_change_dropout_rate
from ea.mutation.pooling_mutations import pool_insert_layer, pool_remove_layer, pool_alter_stride, \
    pool_alter_kernel_size, pool_change_padding
from ea.mutation.dropout_mutations import dropout_add_layer, dropout_remove_layer, dropout_alter_rate
import config as cfg


# Mutate the given DNA
def mutate_dna(dna: DNA):
    # Pick a random mutation
    pick = choice(cfg.allowed_mutations)
    if cfg.DEBUG:
        print("Mutation on this DNA:" + pick)
    # Select mutation based on pick

    # Dense mutations
    if pick == 'dense_insert_layer':
        dense_insert_layer(dna=dna)
    elif pick == 'dense_remove_layer':
        dense_remove_layer(dna=dna)
    elif pick == 'dense_inc_nbr_neurons':
        dense_inc_nbr_of_neurons(dna=dna)
    elif pick == 'dense_dec_nbr_neurons':
        dense_dec_nbr_of_neurons(dna=dna)
    elif pick == 'dense_alter_activation':
        dense_alter_activation_function(dna=dna)
    elif pick == 'dense_flip_useDropout':
        dense_flip_dropout_use(dna=dna)
    elif pick == 'dense_change_dropout_rate':
        dense_change_dropout_rate(dna=dna)

    # Conv mutations
    elif pick == 'conv_insert_layer':
        conv_insert_layer(dna=dna)
    elif pick == 'conv_remove_layer':
        conv_remove_layer(dna=dna)
    elif pick == 'conv_alter_stride':
        conv_alter_stride(dna=dna)
    elif pick == 'conv_alter_nbr_filters':
        conv_alter_filter_nbr(dna=dna)
    elif pick == 'conv_alter_filter_size':
        conv_alter_filter_size(dna=dna)
    elif pick == 'conv_change_padding':
        conv_change_padding(dna=dna)
    elif pick == 'conv_alter_activation':
        conv_alter_activation_function(dna=dna)
    elif pick == 'conv_flip_useBN':
        conv_flip_bn_use(dna=dna)
    elif pick == 'conv_flip_useDropout':
        conv_flip_dropout_use(dna=dna)
    elif pick == 'conv_change_dropout_rate':
        conv_change_dropout_rate(dna=dna)

    # Pooling mutations
    elif pick == 'pool_insert_layer':
        pool_insert_layer(dna=dna)
    elif pick == 'pool_remove_layer':
        pool_remove_layer(dna=dna)
    elif pick == 'pool_alter_stride':
        pool_alter_stride(dna=dna)
    elif pick == 'pool_alter_kernel_size':
        pool_alter_kernel_size(dna=dna)
    elif pick == 'pool_change_padding':
        pool_change_padding(dna=dna)

    # Dropout mutations
    elif pick == 'dropout_add_layer':
        dropout_add_layer(dna=dna)
    elif pick == 'dropout_remove_layer':
        dropout_remove_layer(dna=dna)
    elif pick == 'dropout_alter_rate':
        dropout_alter_rate(dna=dna)

    # Default
    else:
        raise MutationError("Error: picked mutation not implemented")


