from dna.DNA import DNA
from dna.dna_blocks import ConvLayer, DenseLayer, PoolingLayer, FlattenLayer, DropoutLayer, BatchNormLayer
import hashlib

# Variable definition
fn_log = './log/arch_log.txt'


def get_architecture_as_string(dna: DNA):
    # Get architecture from DNA an generate a string based on it
    tmp_str = ""
    for layer in dna.architecture:
        # Write a letter based on the layer type
        if type(layer) is ConvLayer:
            tmp_str = tmp_str + "___C"
        elif type(layer) is DenseLayer:
            tmp_str = tmp_str + "___D"
        elif type(layer) is PoolingLayer:
            tmp_str = tmp_str + "___P"
        elif type(layer) is FlattenLayer:
            tmp_str = tmp_str + "___F"
        elif type(layer) is DropoutLayer:
            tmp_str = tmp_str + "___Drop"
        elif type(layer) is BatchNormLayer:
            tmp_str = tmp_str + "___BN"
        # Write the layer details
        tmp_str = tmp_str + str(layer.__dict__)
    # Return the string
    return tmp_str


def get_hash_from_string(string: str):
    # Generate a hash value from the string
    hash_value = hashlib.sha256(string.encode()).hexdigest()
    return hash_value


def get_hash_from_dna(dna: DNA):
    # Get the architecture string
    arch_str = get_architecture_as_string(dna=dna)
    # Generate a hash value from the string
    hash_value = get_hash_from_string(string=arch_str)
    return hash_value


class ArchLog(object):
    # Init function
    def __init__(self, restart=False):
        # Create a new dict to log the hash values
        self.arch_archive = set()
        # Check if new or restart
        if not restart:
            # Create a new log entry
            with open(fn_log, 'a') as myfile:
                myfile.write("\n")
                myfile.write("-------------------------------------------------------\n")
        else:
            # Load old file and parse entries into dict
            with open(fn_log, 'r') as myfile:
                for line in myfile:
                    # Get an architecture string and log it
                    arch_str = line.strip().split(sep=':', maxsplit=1)
                    if len(arch_str) == 2:
                        print(arch_str[1])
                        # Get the hash of the string
                        new_hash = get_hash_from_string(string=arch_str[1])
                        # Log the hash into the archive
                        self.arch_archive.add(new_hash)

    def log_architecture(self, dna: DNA, debug=False):
        # Get the archtitecture string
        arch_str = get_architecture_as_string(dna=dna)
        # Log the architecture string into the file
        with open(fn_log, 'a') as myfile:
            myfile.write("ID {}:".format(dna.dna_ID))
            myfile.write(arch_str)
            myfile.write("\n")
        # Get the hash of the string
        new_hash = get_hash_from_string(string=arch_str)
        # Log the hash into the archive
        self.arch_archive.add(new_hash)
        if True:
            print("DNA {} has hash {}".format(dna.dna_ID, new_hash))

    def check_if_duplicate_solution(self, dna: DNA):
        # Load the hash of the architecture
        tmp_hash = get_hash_from_dna(dna=dna)
        # Check if it is already in the archive (True = Yes)
        return tmp_hash in self.arch_archive
