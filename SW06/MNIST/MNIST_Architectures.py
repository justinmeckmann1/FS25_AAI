from dna.dna_blocks import ConvLayer, DenseLayer, PoolingLayer, FlattenLayer, DropoutLayer


# Reference model for MNIST - example from Kaggle
# https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
def getMNISTReferenceArchitecture_Kaggle():
    # Generate new empty architecture
    arch = list()

    # Layer 1 - Conv
    tmp_layer = ConvLayer(filternbr=32, kernelsize=5, stride=1, padding='same',
                          usebias=True, activation='relu', useBN=False)
    list.append(arch, tmp_layer)

    # Layer 2 - Conv
    tmp_layer = ConvLayer(filternbr=32, kernelsize=5, stride=1, padding='same',
                          usebias=True, activation='relu', useBN=False)
    list.append(arch, tmp_layer)

    # Layer 3 - Pooling
    tmp_layer = PoolingLayer(poolsize=2, stride=2, padding='valid')
    list.append(arch, tmp_layer)

    # Layer 4 - Dropout
    tmp_layer = DropoutLayer(rate=0.25)
    list.append(arch, tmp_layer)

    # Layer 5 - Conv
    tmp_layer = ConvLayer(filternbr=64, kernelsize=3, stride=1, padding='same',
                          usebias=True, activation='relu', useBN=False)
    list.append(arch, tmp_layer)

    # Layer 6 - Conv
    tmp_layer = ConvLayer(filternbr=64, kernelsize=3, stride=1, padding='same',
                          usebias=True, activation='relu', useBN=False)
    list.append(arch, tmp_layer)

    # Layer 7 - Pooling
    tmp_layer = PoolingLayer(poolsize=2, stride=2, padding='valid')
    list.append(arch, tmp_layer)

    # Layer 8 - Dropout
    tmp_layer = DropoutLayer(rate=0.25)
    list.append(arch, tmp_layer)

    # Flattening layer
    tmp_layer = FlattenLayer()
    list.append(arch, tmp_layer)

    # Layer 9 - Dense
    tmp_layer = DenseLayer(neuronnbr=256, usebias=True, activation='relu', useDropOut=False)
    list.append(arch, tmp_layer)

    # Layer 10 - Dropout
    tmp_layer = DropoutLayer(rate=0.5)
    list.append(arch, tmp_layer)

    # Layer 11 - Dense
    tmp_layer = DenseLayer(neuronnbr=10, usebias=True, activation='softmax', useDropOut=False)
    list.append(arch, tmp_layer)

    return arch

# Reference model for MNIST - example from Keras
# https://keras.io/examples/vision/mnist_convnet/
def getMNISTReferenceArchitecture_Keras():
    # Generate new empty architecture
    arch = list()

    # Layer 1 - Conv
    tmp_layer = ConvLayer(filternbr=32, kernelsize=3, stride=1, padding='valid',
                          usebias=True, activation='relu', useBN=False)
    list.append(arch, tmp_layer)

    # Layer 2 - Pooling
    tmp_layer = PoolingLayer(poolsize=2, stride=2, padding='valid')
    list.append(arch, tmp_layer)

    # Layer 3 - Conv
    tmp_layer = ConvLayer(filternbr=64, kernelsize=3, stride=1, padding='valid',
                          usebias=True, activation='relu', useBN=False)
    list.append(arch, tmp_layer)

    # Layer 4 - Pooling
    tmp_layer = PoolingLayer(poolsize=2, stride=2, padding='valid')
    list.append(arch, tmp_layer)

    # Flattening layer
    tmp_layer = FlattenLayer()
    list.append(arch, tmp_layer)

    # Layer 10 - Dropout
    tmp_layer = DropoutLayer(rate=0.5)
    list.append(arch, tmp_layer)

    # Layer 11 - Dense
    tmp_layer = DenseLayer(neuronnbr=10, usebias=True, activation='softmax', useDropOut=False)
    list.append(arch, tmp_layer)

    return arch