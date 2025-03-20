from dna.DNA import DNA
from dna.dna_blocks import ConvLayer, DenseLayer, PoolingLayer, FlattenLayer, DropoutLayer
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, Softmax, BatchNormalization


def createKerasModelFromArchitecture(dna: DNA, debug=False):
    # Get the number of layers in the network
    nbr_of_layers = len(dna.architecture)

    # Prepare a sequential Keras model
    model = Sequential()

    # Loop through all layers from start to end
    i = 1
    for layer in dna.architecture:
        # Check layer type

        # Convolution layer
        if isinstance(layer, ConvLayer):
            # Layer
            if i == 1:
                model.add(Conv2D(filters=layer.nbr_of_filters, kernel_size=layer.kernel_size,
                                 strides=layer.stride, padding=layer.padding, use_bias=layer.use_bias,
                                 input_shape=dna.static_input_shape))
            else:
                model.add(Conv2D(filters=layer.nbr_of_filters, kernel_size=layer.kernel_size,
                                 strides=layer.stride, padding=layer.padding, use_bias=layer.use_bias))
            # Activation
            if layer.activation_type == 'linear':
                pass
            elif layer.activation_type == 'relu':
                model.add(Activation('relu'))
            else:
                print("Error: Activation type unknown: " + str(layer.activation_type))
            # Batch normalization
            if layer.hasBN:
                model.add(BatchNormalization())
            # Dropout
            if layer.hasDropOut:
                model.add(Dropout(rate=layer.dropout_rate))

        # Dense layer
        elif isinstance(layer, DenseLayer):
            # Layer
            if i == 1:
                model.add(Dense(units=layer.nbr_of_neurons, use_bias=layer.use_bias,
                                input_shape=dna.static_input_shape))
            else:
                model.add(Dense(units=layer.nbr_of_neurons, use_bias=layer.use_bias))
            # Activation
            if layer.activation_type == 'linear':
                pass
            elif layer.activation_type == 'relu':
                model.add(Activation('relu'))
            elif layer.activation_type == 'softmax':
                model.add(Softmax())
            else:
                print("Error: Activation type unknown: " + str(layer.activation_type))
            # Dropout
            if layer.hasDropOut:
                model.add(Dropout(rate=layer.dropout_rate))

        # Pooling layer
        elif isinstance(layer, PoolingLayer):
            if i == 1:
                model.add(MaxPooling2D(pool_size=layer.pool_size, strides=layer.stride,
                                       padding=layer.padding, input_shape=dna.static_input_shape))
            else:
                model.add(MaxPooling2D(pool_size=layer.pool_size, strides=layer.stride,
                                       padding=layer.padding))

        # Dropout layer
        elif isinstance(layer, DropoutLayer):
            if i == 1:
                model.add(Dropout(rate=layer.dropout_rate, input_shape=dna.static_input_shape))
            else:
                model.add(Dropout(rate=layer.dropout_rate))

        # # Batch normalization layer
        # elif isinstance(layer, BatchNormLayer):
        #     if i == 1:
        #         model.add(BatchNormalization(input_shape=dna.static_input_shape))
        #     else:
        #         model.add(BatchNormalization())

        # Flattening layer
        elif isinstance(layer, FlattenLayer):
            if i == 1:
                model.add(Flatten(input_shape=dna.static_input_shape))
            else:
                model.add(Flatten())

        # Unknown layer
        else:
            print("Error creating model from architecture. Layer type unknown: " + str(type(layer)))

        # Increase i
        i = i + 1

    # Print summary
    if debug:
        model.summary()


    # Set reference in DNA
    dna.keras_model_ref = model

    # Clean up
    del model, layer, i, nbr_of_layers