from dna.DNA import DNA
from ea.sorting_v2 import sort_dna_list_by_criteria


def ranking_macs(population_set):
    # Ranking
    pop_list = list(population_set)

    # Sort list by number of macs
    sort_dna_list_by_criteria(dna_list=pop_list, criteria="macs")

    # Set the rank of each DNA in the list
    n = len(pop_list)
    for dna in pop_list:
        assert isinstance(dna, DNA)
        dna.setRank(new_rank=n)
        n = n-1


def ranking_luts(population_set):
    # Ranking
    pop_list = list(population_set)

    # Sort list by number of luts
    sort_dna_list_by_criteria(dna_list=pop_list, criteria="luts")

    # Set the rank of each DNA in the list
    n = len(pop_list)
    for dna in pop_list:
        assert isinstance(dna, DNA)
        dna.setRank(new_rank=n)
        n = n-1


def ranking_acc(population_set):
    # Ranking
    pop_list = list(population_set)

    # Sort list by number of macs
    sort_dna_list_by_criteria(dna_list=pop_list, criteria="acc")

    # Set the rank of each DNA in the list
    n = len(pop_list)
    for dna in pop_list:
        assert isinstance(dna, DNA)
        dna.setRank(new_rank=n)
        n = n-1


def ranking_luts_and_difference_by_complexity(population_set):
    # Ranking
    pop_list = list(population_set)

    # Sort list by number of macs
    sort_dna_list_by_criteria(dna_list=pop_list, criteria="diff")

    # Set the rank of each DNA in the list
    n = len(pop_list)
    for dna in pop_list:
        assert isinstance(dna, DNA)
        dna.setRank(new_rank=n)
        n = n-1


def ranking_cc(population_set):
    # Ranking
    pop_list = list(population_set)

    # Sort list by number of macs
    sort_dna_list_by_criteria(dna_list=pop_list, criteria="cc")

    # Set the rank of each DNA in the list
    n = len(pop_list)
    for dna in pop_list:
        assert isinstance(dna, DNA)
        dna.setRank(new_rank=n)
        n = n-1


def ranking_by_criteria(population_set, criteria):
    # Ranking
    pop_list = list(population_set)

    # Sort list by number of macs
    sort_dna_list_by_criteria(dna_list=pop_list, criteria=criteria)

    # Set the rank of each DNA in the list
    n = len(pop_list)
    for dna in pop_list:
        assert isinstance(dna, DNA)
        dna.setRank(new_rank=n)
        n = n-1