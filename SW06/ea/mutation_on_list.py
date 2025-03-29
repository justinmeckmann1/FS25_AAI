from dna.DNA import DNA
from dna.createModelFromArch import createKerasModelFromArchitecture
from ea.mutation.mutation_helper import MutationError
from ea.mutation.mutation_on_dna import mutate_dna
import traceback
import sys


def mutate_dna_list(parent_dna_list, child_set, ssc, nbr_of_children_per_parent=1, round_nbr=0, debug=False):
    # Iterate over the DNAs in the list
    for dna in parent_dna_list:
        assert isinstance(dna, DNA)

        for i in range(nbr_of_children_per_parent):
            # create a child (copy parent and mutate copy)
            while True:
                # Create a copy of the parent
                childDNA = dna.getCopy()

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
                    if ssc.check_if_dna_allowed(dna=childDNA):
                        # if we reach this point, no error occured and we can break out of the loop
                        break
                    else:
                        if debug:
                            print("Search space constrains violated. Try new mutation.")
                            del childDNA
                        continue
                except MutationError:
                    del childDNA
                    if debug:
                        print("Mutation error")
                    continue
                except ValueError:
                    # Error when architecture has a mismatch during tensorflow creation
                    # e.g. Negative dimension size caused by subtracting 9 from 8 for op: 'MaxPool' (shape: [?,8,8,20])
                    del childDNA
                    if debug:
                        print("Value error")
                    continue
                except Exception as e:
                    print("Error while creating childDNA: {}".format(str(e)))
                    traceback.print_exc(file=sys.stdout)
                    del childDNA
                    continue

            # Give it an ID
            childDNA.setDNA_ID()

            # Set the generation number
            childDNA.generation = round_nbr+1

            # Set number of total mutations
            childDNA.nbr_of_total_mutations = dna.nbr_of_total_mutations + 1

            # Log the architecture
            ssc.static_arch_log_ref.log_architecture(dna=childDNA, debug=debug)

            # Add it to the active set
            child_set.add(childDNA)

            # Clean up
            del childDNA
        del dna
