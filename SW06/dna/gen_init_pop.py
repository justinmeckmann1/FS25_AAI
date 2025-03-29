from dna.DNA import DNA
from dna.createModelFromArch import createKerasModelFromArchitecture
from numpy import random
from ssc.search_space_constraints import SearchSpaceConstrains
from ea.mutation.mutation_helper import MutationError
from ea.mutation.mutation_on_dna import mutate_dna
import config as cfg


def generateInitialPopulation(ground_dna: DNA, ssc: SearchSpaceConstrains, population_size: int=50, debug=False):
    # Create a set to hold the active DNA objects
    dna_set = set()

    # Create a keras model from the basic architecture
    createKerasModelFromArchitecture(dna=ground_dna, debug=True)

    # Check it for logging purpose
    ssc.check_if_dna_allowed(dna=ground_dna, debug=True)
    # Log the architecture
    ssc.static_arch_log_ref.log_architecture(dna=ground_dna, debug=debug)
    # Add basic DNA to the active DNA set
    dna_set.add(ground_dna)

    # Create a population of different DNAs
    for i in range(population_size-1):
        # Parent selection
        if cfg.ground_dna_is_always_parent:
            parent_DNA = ground_dna
        else:
            # Pick a parent randomly (uniform distribution) from the set
            parent_DNA = random.choice(list(dna_set))

        assert isinstance(parent_DNA, DNA)
        if debug:
            print("Parent DNA picked:", str(parent_DNA.dna_ID))

        # create a child (copy parent and mutate copy)
        while True:
            # Create a copy of the parent
            childDNA = parent_DNA.getCopy()

            try:
                # Mutate the copy
                mutate_dna(dna=childDNA)

                # Check for duplicate architectures
                if ssc.static_arch_log_ref.check_if_duplicate_solution(dna=childDNA):
                    # Is a duplicate, so we throw an error
                    print("Duplicate detected")
                    raise MutationError("Error: generated architecture already in history log")

                # Create a keras model from the architecture
                createKerasModelFromArchitecture(dna=childDNA, debug=debug)

                # Check if the child fulfills the search space constrains
                if ssc.check_if_dna_allowed(dna=childDNA, debug=debug):
                    # if we reach this point, no error occured and we can break out of the loop
                    break
                else:
                    if debug:
                        print("Search space constrains violated. Try new mutation.")
                        del childDNA
                    continue
            except MutationError:
                if debug:
                    print("Mutation error")
                del childDNA
                continue
            except ValueError:
                # Error when architecture has a mismatch during tensorflow creation
                # e.g. Negative dimension size caused by subtracting 9 from 8 for op: 'MaxPool' (shape: [?,8,8,20])
                if debug:
                    print("Value error")
                del childDNA
                continue

        # Give it an ID
        childDNA.setDNA_ID()

        # Set nbr of total mutations
        childDNA.nbr_of_total_mutations = 1

        # Log the architecture
        ssc.static_arch_log_ref.log_architecture(dna=childDNA, debug=debug)

        # Add it to the active set
        dna_set.add(childDNA)
        if debug:
            print("Child creation complete. DNA:", str(childDNA.dna_ID))

    # Return the initial population
    return dna_set