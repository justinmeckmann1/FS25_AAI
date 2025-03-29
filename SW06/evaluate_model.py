import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse




import config as cfg
from MNIST.Usecase_MNIST import MNISTUseCase
from CIFAR.Usecase_CIFAR import CIFARUseCase

def evaluate_model(model_file):

    # Model to evaluate
    # model_file  = "best_model_dna_3.hdf5" # Update with your model path

    # Load trained model from H5 file
    model_path = "log/training_models/" 
    model_name = model_path + model_file + ".h5"
    model = keras.models.load_model(model_name)

    model.summary()

    # Load test dataset depending on usecase
    if cfg.USECASE == "MNIST":
        usecase = MNISTUseCase(debug=cfg.DEBUG)
    elif cfg.USECASE == "CIFAR": 
        usecase = CIFARUseCase(debug=cfg.DEBUG)

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(usecase.testing_data, verbose=1)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Optional: Retrain the model
    retrain = input("Do you want to retrain the model? (yes/no): ").strip().lower()
    if retrain == "yes":
        print(f"Start retraining with {cfg.nbr_of_training_epochs:d} epochs....")
        model.fit(usecase.training_data, 
                    validation_data=usecase.validation_data,
                    epochs=cfg.nbr_of_training_epochs, 
                    verbose=2)

        # Save the updated model
        model_name = model_path + model_file + "_retrnd.h5"
        model.save(model_name)
        print("Model retrained and saved as " + model_name)

        # Re-evaluate the model on the test set
        print("Evaluate retrained model on testset ....")
        loss, accuracy = model.evaluate(usecase.testing_data, verbose=0)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a CNN model and perform optional retraining.")
    parser.add_argument("name", type=str, help="Model name, e.g. best_model_dna_3")
    args = parser.parse_args()
    
    print(f"Evaluating model: {args.name}")
    
    evaluate_model(model_file=args.name)

if __name__ == "__main__":
    main()