import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse
import tf2onnx
import onnx
from PIL import Image as im 

import config as cfg
from MNIST.Usecase_MNIST import MNISTUseCase
from CIFAR.Usecase_CIFAR import CIFARUseCase

def export_model(model_file, batch_size):

    # Load trained model from H5 file
    model_path = "log/training_models/" 
    model_name = model_path + model_file + ".h5"
    model = keras.models.load_model(model_name)

    model.summary()

    # Extract dimensions from model input
    _, height, width, channels = model.input_shape  # NHWC format
 
    # Define the input signature in NHWC
    input_signature = [tf.TensorSpec([batch_size, height, width, channels], tf.float32, name='image')]
 
    # Convert the model to ONNX using NCHW layout
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        inputs_as_nchw=["image"]
    )

    # Save the converted model
    model_name = model_path + model_file + ".onnx"
    onnx.save(onnx_model, model_name)
    print("Model converted and saved as " + model_name)

    #### from the testset generate numpy arrays for later testing
        # Load test dataset depending on usecase
    if cfg.USECASE == "MNIST":
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    elif cfg.USECASE == "CIFAR": 
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # confine testset to the number of images to be used
    x_test = x_test[:batch_size]

    # float-version of test input
    x_test_float = x_test.astype("float32") / 255

    # perform predictions with this data to generate expected results
    predictions = model.predict(x_test_float, batch_size = batch_size, steps = 1)
    print(predictions.shape)

    # convert test input from channel-last to channel-first representation
    x_test_cf = np.moveaxis(x_test, 3, 1)
    x_test_cf_float = np.moveaxis(x_test_float, 3, 1)

    # verify channel-first conversion by showing the RED color-channel of the original channel-last version and the re-verted channel-first converted version of the last image in the seleted batch.
    image1 = im.fromarray(x_test[batch_size-1,:,:,0]) 
    #image1.show()
    image2 = im.fromarray(np.moveaxis(x_test_cf, 1, 3)[batch_size-1,:,:,0]) 
    #image2.show()

    # save test input and expected result as numpy arrays
    np.save(model_path + model_file + "_input.npy", x_test_cf_float)
    np.save(model_path + model_file + "_output.npy", predictions)

def main():
    parser = argparse.ArgumentParser(description="Export a trained CNN model in ONNX format.")
    parser.add_argument("name", type=str, help="Model name, e.g. best_model_dna_6")
    parser.add_argument("batch", type=int, help="Batch size, e.g. 10")
    args = parser.parse_args()
    
    print(f"Exporting model: {args.name}")
    
    export_model(model_file=args.name, batch_size=args.batch)

if __name__ == "__main__":
    main()