import copy
import config as cfg


class DNA(object):
    """DNA definition"""
    # static (class) variable
    static_DnaID_counter = int(0)
    static_input_shape = 0

    # Creation of a new DNA
    def __init__(self):
        self.dna_ID = -1
        self.parent_ID = -1
        self.architecture = []
        self.keras_model_ref = None
        self.learning_rate = cfg.init_LR
        self.nbr_of_parameters = 0
        self.nbr_of_macs = 0
        self.nbr_of_luts = 0
        self.nbr_of_luts_halfM = 0
        self.max_acc_reached = 0.0
        self.max_acc_reached_halfM = 0.0
        self.diff_high_low_compl = 0.0
        self.fitness = 0.0
        self.param_rank_value_for_sort = 0.0
        self.rank = 0
        self.generation = 0
        self.nbr_of_total_mutations = 0

    # Get the total number of DNAs generated
    @classmethod
    def getNbrOfDnaTotal(cls):
        return cls.static_DnaID_counter

    # Get a new DNA ID number
    @classmethod
    def getNewDnaIDnbr(cls):
        cls.static_DnaID_counter = cls.static_DnaID_counter + 1
        return cls.static_DnaID_counter

    @classmethod
    def setInputShape(cls, shape):
        cls.static_input_shape = shape

    # Set an architecture
    def setArchitecture(self, architecture):
        self.architecture = architecture

    # Get an architecture
    def getArchitecture(self):
        return self.architecture

    # Set a DNA ID
    def setDNA_ID(self, id_number: int = -1):
        if id_number == -1:
            # automatically enumerate
            self.dna_ID = (self.getNewDnaIDnbr()) - 1
        else:
            self.dna_ID = id_number

    # Set the number of parameters
    def setNbrOfParameters(self, nbrOfParam: int):
        self.nbr_of_parameters = nbrOfParam

    # Set the number of MACs
    def setNbrOfMacs(self, nbrOfMacs: int):
        self.nbr_of_macs = nbrOfMacs

    # Set the number of LUTs high complexity
    def setNbrOfLuts(self, nbrOfLuts: int):
        self.nbr_of_luts = nbrOfLuts

    # Set the number of LUTs low complexity
    def setNbrOfLuts_halfM(self, nbrOfLuts: int):
        self.nbr_of_luts_halfM = nbrOfLuts

    # Set the maximum accuracy reached during testing for the high complexity
    def setMaxAccReached(self, maxAcc):
        self.max_acc_reached = float(maxAcc)

     # Set the maximum accuracy reached during testing for the low complexity
    def setMaxAccReached_half_M(self, maxAcc):
        self.max_acc_reached_halfM = float(maxAcc)

    # Get the maximum accuracy reached during testing for the high complexity
    def getMaxAccReached(self):
        return self.max_acc_reached

     # Get the maximum accuracy reached during testing for the low complexity
    def getMaxAccReached_low_compl(self):
        return self.max_acc_reached_halfM

    # Set the diff accuracy for high and low complexity
    def setDiff_high_low_compl(self):
        self.diff_high_low_compl = self.max_acc_reached - self.max_acc_reached_halfM

    # Get the diff accuracy for high and low complexity
    def getDiff_high_low_compl(self):
        return self.diff_high_low_compl

    # Set the fitness value of the DNA
    def setFitness(self, fitness: float):
        self.fitness = fitness

    # Set the rank of the DNA
    def setRank(self, new_rank: int):
        self.rank = new_rank

    # Get a copy of the DNA (parent ID, architecture and learning rate)
    def getCopy(self):
        # Make a new DNA
        newDNA = DNA()
        # Set the parent ID
        newDNA.parent_ID = self.dna_ID
        # Copy the architecture
        newDNA.setArchitecture(architecture=copy.deepcopy(self.getArchitecture()))
        # Copy the learning rate
        newDNA.learning_rate = self.learning_rate
        return newDNA