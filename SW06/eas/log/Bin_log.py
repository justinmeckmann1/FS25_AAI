import datetime
from dna.DNA import DNA
import numpy as np

# Variable definition
fn_log = './log/bin_log.txt'


# Init function
def init_dna_log():
    # Create a new log entry
    with open(fn_log, 'a') as myfile:
        myfile.write("\n")
        myfile.write("-------------------------------------------------------\n")
        myfile.write("-------------------------------------------------------\n")
        myfile.write("-------------------------------------------------------\n")
        myfile.write("New Binary Log: ")
        myfile.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        myfile.write("\n")
        myfile.write("dna_id,parent_dna_id,generation,nbr_total_mutations")
        myfile.write(",max_val_acc,nbr_of_macs,nbr_of_param,nbr_of_epochs\n")


def add_dna_to_log(dna: DNA, debug=False):
    # Extract data from training
    val_acc_list = dna.keras_model_ref.history.history['val_categorical_accuracy']
    val_acc = np.max(val_acc_list)
    nbr_of_training_epochs = len(val_acc_list)

    # Log the best accuracy into the dna
    dna.setMaxAccReached(maxAcc=val_acc)

    # Log the data into the log file
    with open(fn_log, 'a') as myfile:
        myfile.write(str(dna.dna_ID) + ',')
        myfile.write(str(dna.parent_ID) + ',')
        myfile.write(str(dna.generation) + ',')
        myfile.write(str(dna.nbr_of_total_mutations) + ',')
        myfile.write(str(val_acc) + ',')
        myfile.write(str(dna.nbr_of_macs) + ',')
        myfile.write(str(dna.nbr_of_parameters) + ',')
        myfile.write(str(nbr_of_training_epochs))
        myfile.write("\n")