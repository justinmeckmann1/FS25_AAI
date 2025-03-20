from dna.DNA import DNA
import pickle


def add_dna_weights_to_log(dna: DNA, debug=False):
    # Log the data into the log file
    temp = dna.keras_model_ref
    afile = open(r'log/training_weights_init_pop/id' + str(dna.dna_ID) + '.pkl', 'wb')
    pickle.dump(temp, afile)
    afile.close()


def load_dna_weights_from_log(dna: DNA, debug=False):
    print("Load CNN from Init Pop with ID:" + str(dna.dna_ID))
    file2 = open(r'log/training_weights_init_pop/id' + str(dna.dna_ID) + '.pkl', 'rb')
    new_d = pickle.load(file2)
    file2.close()
    return new_d
