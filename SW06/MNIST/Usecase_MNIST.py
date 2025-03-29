from keras.utils import to_categorical
from MNIST.MNIST_Architectures import getMNISTReferenceArchitecture_Kaggle, getMNISTReferenceArchitecture_Keras
import tensorflow.keras as keras
import tensorflow as tf
import config as cfg
import numpy as np
import h5py
from sklearn.model_selection import train_test_split

num_classes = 10


def get_usecase_data(batch_size, debug=False):
    # Load training data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Split again into training and validation data
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=True)

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_val = x_val.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Check dataset size valid
    if cfg.train_dataset_size > x_train.shape[0]:
        cfg.train_dataset_size = x_train.shape[0]
    if cfg.val_dataset_size > x_val.shape[0]:
        cfg.val_dataset_size = x_val.shape[0]

    # Decrease training data size
    x_train = x_train[:cfg.train_dataset_size,:,:]
    y_train = y_train[:cfg.train_dataset_size]
    x_val = x_val[:cfg.val_dataset_size,:,:]
    y_val = y_val[:cfg.val_dataset_size]


    # Make sure images have correct shape
    x_train = np.expand_dims(x_train, -1)
    x_val = np.expand_dims(x_val, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)

    print(x_train.shape[0], "train samples")
    unique, counts = np.unique(y_train, return_counts=True)
    print("Balance: ", dict(zip(unique, counts)))

    print(x_val.shape[0], "val samples")
    unique, counts = np.unique(y_val, return_counts=True)
    print("Balance: ", dict(zip(unique, counts)))

    print(x_test.shape[0], "test samples")
    unique, counts = np.unique(y_test, return_counts=True)
    print("Balance: ", dict(zip(unique, counts)))

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Create tf.data.Dataset for efficient data loading and augmentation
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Optimize data pipeline
    def preprocess_data(image, label):
        if cfg.use_augmentation:
            image = tf.image.random_shift(image, 0.1, 0.1)
            image = tf.image.random_zoom(image, 0.1)
            image = tf.image.random_rotate(image, 0.2)
        return image, label

    # Apply preprocessing, batch, shuffle, and prefetch to the datasets
    train_dataset = (
        train_dataset
        .map(preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # Optional: add data augmentations in preprocess_data
        .shuffle(buffer_size=10000)  # Shuffle data to prevent overfitting
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)  # Prefetch next batch during training
    )

    val_dataset = (
        val_dataset
        .map(preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    test_dataset = (
        test_dataset
        .map(preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    # Return all datasets
    return train_dataset, val_dataset, test_dataset



class MNISTUseCase(object):
    def __init__(self, debug=False):
        # Prepare the usecase data
        train_dataset, val_dataset, test_dataset = get_usecase_data(cfg.batch_size, debug=debug)
        # Split the usecase data into the right labels
        self.training_data = train_dataset
        self.validation_data = val_dataset
        self.testing_data = test_dataset

        # Define the input shape from train dataset
        self.input_shape = train_dataset.element_spec[0].shape[1:4]  # (28, 28, 1)

    def get_reference_model(self, arch):
        if arch == "Kaggle":
            return getMNISTReferenceArchitecture_Kaggle()
        if arch == "Keras":
            return getMNISTReferenceArchitecture_Keras()
        else:
            print("ERROR: Architecture {} unknown".format(arch))
