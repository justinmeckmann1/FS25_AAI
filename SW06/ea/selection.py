from dna.DNA import DNA
from numpy.random import random


def selection(population_set, candidates_to_pick: int, debug=False):
    # Create a list of the population
    pop_list = list(population_set)

    # Sort list by fitness
    pop_list.sort(key=lambda x: x.fitness, reverse=True)

    # Make a list with the cumulative fitness values (order corresponds to sorted pop_list)
    cumSum_prop_list = list()
    tmp_sum = 0.0
    n = len(pop_list)
    for dna in pop_list:
        assert isinstance(dna, DNA)
        sum = dna.fitness + tmp_sum
        cumSum_prop_list.append(sum/n)  # Divide by n to map between zero and one
        tmp_sum = sum

    # Calculate the distance of the picking spacers
    distance = 1 / candidates_to_pick
    if candidates_to_pick > n:
        print("Error: tried to pick more candidates then there are in the population set.")

    # Pick a random number between zero and distance
    random_point = random() * distance

    # Create a new list of DNAs that get picked
    # Must be a list and not a set, as the same DNA object can be picked multiple times
    new_pop = list()
    index = 0
    for i in range(candidates_to_pick):
        # Calculate number
        nbr = random_point + i*distance
        # Find index of matching DNA
        while True:
            if nbr <= cumSum_prop_list[index]:
                # nbr is equal or smaller then the current segment so we have a hit and
                # break out of the loop
                break
            else:
                # nbr is bigger then this segment end so we continue with the next
                index = index + 1
                continue
        # Insert the DNA at the given indes into the new set
        new_pop.append(pop_list[index])

    # Return the newly created set
    return new_pop
