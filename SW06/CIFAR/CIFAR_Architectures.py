from dna.dna_blocks import ConvLayer, DenseLayer, PoolingLayer, FlattenLayer, DropoutLayer

# Reference model for CIFAR-10 - example from Kaggle
# https://www.kaggle.com/code/faressayah/cifar-10-images-classification-using-cnns-88
def getCIFARReferenceArchitecture_Kaggle():
    # Generate new empty architecture
    arch = list()

    # Layer 1 - Conv
    tmp_layer = ConvLayer(filternbr=32, kernelsize=3, stride=1, padding='same',
                          usebias=True, activation='relu', useBN=True)
    list.append(arch, tmp_layer)

    # Layer 2 - Conv
    tmp_layer = ConvLayer(filternbr=32, kernelsize=3, stride=1, padding='same',
                          usebias=True, activation='relu', useBN=True)
    list.append(arch, tmp_layer)

    # Layer 3 - Pooling
    tmp_layer = PoolingLayer(poolsize=2, stride=2, padding='same')
    list.append(arch, tmp_layer)

    # Layer 4 - Dropout
    tmp_layer = DropoutLayer(rate=0.25)
    list.append(arch, tmp_layer)

    # Layer 5 - Conv
    tmp_layer = ConvLayer(filternbr=64, kernelsize=3, stride=1, padding='same',
                          usebias=True, activation='relu', useBN=True)
    list.append(arch, tmp_layer)

    # Layer 6 - Conv
    tmp_layer = ConvLayer(filternbr=64, kernelsize=3, stride=1, padding='same',
                          usebias=True, activation='relu', useBN=True)
    list.append(arch, tmp_layer)

    # Layer 7 - Pooling
    tmp_layer = PoolingLayer(poolsize=2, stride=2, padding='same')
    list.append(arch, tmp_layer)

    # Layer 8 - Dropout
    tmp_layer = DropoutLayer(rate=0.25)
    list.append(arch, tmp_layer)

    # Layer 9 - conv
    tmp_layer = ConvLayer(filternbr=128, kernelsize=3, stride=1, padding='same',
                          usebias=True, activation='relu', useBN=True)
    list.append(arch, tmp_layer)

    # Layer 10 - conv
    tmp_layer = ConvLayer(filternbr=128, kernelsize=3, stride=1, padding='same',
                          usebias=True, activation='relu', useBN=True)
    list.append(arch, tmp_layer)

    # Layer 11 - Pooling
    tmp_layer = PoolingLayer(poolsize=2, stride=2, padding='same')
    list.append(arch, tmp_layer)

    # Layer 12 - Dropout
    tmp_layer = DropoutLayer(rate=0.25)
    list.append(arch, tmp_layer)

    # Flattening layer
    tmp_layer = FlattenLayer()
    list.append(arch, tmp_layer)

    # Layer 13 - Dense
    tmp_layer = DenseLayer(neuronnbr=128, usebias=True, activation='relu', useDropOut=False)
    list.append(arch, tmp_layer)

    # Layer 14 - Dropout
    tmp_layer = DropoutLayer(rate=0.25)
    list.append(arch, tmp_layer)

    # Layer 15 - Dense
    tmp_layer = DenseLayer(neuronnbr=10, usebias=True, activation='softmax', useDropOut=False)
    list.append(arch, tmp_layer)

    return arch