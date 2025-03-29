import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse
import tf2onnx
import onnx

import config as cfg
from MNIST.Usecase_MNIST import MNISTUseCase
from CIFAR.Usecase_CIFAR import CIFARUseCase

def export_model(model_file):

    # Load trained model from H5 file
    model_path = "log/training_models/" 
    model_name = model_path + model_file + ".h5"
    model = keras.models.load_model(model_name)

    model.summary()

    # Convert the model to onnx
    onnx_model, _ = tf2onnx.convert.from_keras(model)

    # Save the converted model
    model_name = model_path + model_file + ".onnx"
    onnx.save(onnx_model, model_name)
    print("Model converted and saved as " + model_name)

def main():
    parser = argparse.ArgumentParser(description="Export a trained CNN model in ONNX format.")
    parser.add_argument("name", type=str, help="Model name, e.g. best_model_dna_6")
    args = parser.parse_args()
    
    print(f"Exporting model: {args.name}")
    
    export_model(model_file=args.name)

if __name__ == "__main__":
    main()