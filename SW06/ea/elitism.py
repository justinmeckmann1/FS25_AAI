from dna.DNA import DNA
from dna.createModelFromArch import createKerasModelFromArchitecture


def elitism(population_set, elite_set, ssc, nbr_of_elites: int, debug=False):
    # Create a list of the population
    pop_list = list(population_set)

    # Sort list by fitness
    pop_list.sort(key=lambda x: x.fitness, reverse=True)

    # Take the best n DNAs and generate new candidates of them
    for i in range(nbr_of_elites):
        parent_dna = pop_list[i]
        assert isinstance(parent_dna, DNA)

        # Generate a child (copy of the parent)
        childDNA = parent_dna.getCopy()

        # Create a keras model from the architecture
        createKerasModelFromArchitecture(dna=childDNA, debug=debug)

        # Check if the child fulfills the search space constrains
        # This is for logging purpose, we already know that the architecture is ok
        ssc.check_if_dna_allowed(dna=childDNA)

        # Give it an ID
        childDNA.setDNA_ID()

        # Add it to the active set
        elite_set.add(childDNA)

        # Clean up
        del parent_dna, childDNA

    del pop_list