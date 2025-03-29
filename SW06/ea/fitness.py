from dna.DNA import DNA


def fitness_assignment(population_set, selection_pressure: float, debug=False):
    # Variable definition for fitness assignment
    n = len(population_set)  # Number of individuals in list

    for dna in population_set:
        assert isinstance(dna, DNA)
        # Get position of individual
        pos = dna.rank
        # Calculate fitness of individual
        tmp_fitness = 2.0 - selection_pressure + 2.0*(selection_pressure-1.0)*((pos-1)/(n-1))
        # Set fitness
        dna.setFitness(fitness=tmp_fitness)
        if debug:
            print("DNA with ID " + str(dna.dna_ID) + " has fitness: " + str(tmp_fitness))