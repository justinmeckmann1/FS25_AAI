from dna.DNA import DNA
from ea.ranking import ranking_macs, ranking_acc, ranking_luts, ranking_luts_and_difference_by_complexity, ranking_cc, ranking_by_criteria
from ea.fitness import fitness_assignment
from ea.reinsertion import reinsert_and_sort

# Testing code for the EA functions

# Create a set of DNAs
dna_set = set()

# Define DNAs
# DNA MACs: 1 - 2 - 3 - 4 - 5 (small -> big)
# DNA Acc:  3 - 2 - 4 - 1 - 5 (good -> bad)

# DNA 1
dna_1 = DNA()
dna_1.dna_ID = 1
dna_1.parent_ID = 0
dna_1.max_acc_reached = 0.85
dna_1.fitness = 0.0
dna_1.rank = 0
dna_1.nbr_of_macs = 10000
dna_1.nbr_of_luts = 50000
dna_1.max_cc_needed = 50000
dna_1.max_acc_reached_halfM = 0.1
dna_1.diff_high_low_compl = 0.5
dna_set.add(dna_1)

# DNA 2
dna_2 = DNA()
dna_2.dna_ID = 2
dna_2.parent_ID = 0
dna_2.max_acc_reached = 0.95
dna_2.fitness = 0.0
dna_2.rank = 0
dna_2.nbr_of_macs = 20000
dna_2.nbr_of_luts = 40000
dna_2.max_cc_needed = 40000
dna_2.max_acc_reached_halfM = 0.2
dna_2.diff_high_low_compl = 0.4
dna_set.add(dna_2)

# DNA 3
dna_3 = DNA()
dna_3.dna_ID = 3
dna_3.parent_ID = 0
dna_3.max_acc_reached = 0.94
dna_3.fitness = 0.0
dna_3.rank = 0
dna_3.nbr_of_macs = 30000
dna_3.nbr_of_luts = 30000
dna_3.max_cc_needed = 30000
dna_3.max_acc_reached_halfM = 0.3
dna_3.diff_high_low_compl = 0.3
dna_set.add(dna_3)

# DNA 4
dna_4 = DNA()
dna_4.dna_ID = 4
dna_4.parent_ID = 0
dna_4.max_acc_reached = 0.90
dna_4.fitness = 0.0
dna_4.rank = 0
dna_4.nbr_of_macs = 40000
dna_4.nbr_of_luts = 20000
dna_4.max_cc_needed = 20000
dna_4.max_acc_reached_halfM = 0.4
dna_4.diff_high_low_compl = 0.2
dna_set.add(dna_4)

# DNA 5
dna_5 = DNA()
dna_5.dna_ID = 5
dna_5.parent_ID = 0
dna_5.max_acc_reached = 0.80
dna_5.fitness = 0.0
dna_5.rank = 0
dna_5.nbr_of_macs = 50000
dna_5.nbr_of_luts = 10000
dna_5.max_cc_needed = 10000
dna_5.max_acc_reached_halfM = 0.5
dna_5.diff_high_low_compl = 0.1
dna_set.add(dna_5)

# Start of test
print("TEST START")

print("Ranking CCs:")
# Test ranking
ranking_by_criteria(population_set=dna_set, criteria="cc")
# Check ranking (dna 1 must have rank 5)
if dna_1.rank != 1:
    print("Ranking error - 1")
if dna_2.rank != 2:
    print("Ranking error - 2")
if dna_3.rank != 3:
    print("Ranking error - 3")
if dna_4.rank != 4:
    print("Ranking error - 4")
if dna_5.rank != 5:
    print("Ranking error - 5")

print("Ranking Macs:")
# Test ranking
ranking_by_criteria(population_set=dna_set, criteria="macs")
# Check ranking (dna 1 must have rank 5)
if dna_1.rank != 5:
    print("Ranking error - 1")
if dna_2.rank != 4:
    print("Ranking error - 2")
if dna_3.rank != 3:
    print("Ranking error - 3")
if dna_4.rank != 2:
    print("Ranking error - 4")
if dna_5.rank != 1:
    print("Ranking error - 5")

print("Ranking Luts:")
# Test ranking
ranking_by_criteria(population_set=dna_set, criteria="luts")
# Check ranking (dna 1 must have rank 5)
if dna_5.rank != 5:
    print("Ranking error - 1")
if dna_4.rank != 4:
    print("Ranking error - 2")
if dna_3.rank != 3:
    print("Ranking error - 3")
if dna_2.rank != 2:
    print("Ranking error - 4")
if dna_1.rank != 1:
    print("Ranking error - 5")

print("Ranking Acc:")
# Test ranking
ranking_by_criteria(population_set=dna_set, criteria="acc")
# Check ranking (dna 1 must have rank 5)
if dna_2.rank != 5:
    print("Ranking error - 1")
if dna_3.rank != 4:
    print("Ranking error - 2")
if dna_4.rank != 3:
    print("Ranking error - 3")
if dna_1.rank != 2:
    print("Ranking error - 4")
if dna_5.rank != 1:
    print("Ranking error - 5")

print("Ranking luts and difference:")
# Test ranking
ranking_by_criteria(population_set=dna_set, criteria="luts")
#ranking_luts_and_difference_by_complexity(population_set=dna_set)
# Check ranking (dna 1 must have rank 5)
if dna_5.rank != 5:
    print("Ranking error - 1")
    print(dna_5.param_rank_value_for_sort)
if dna_4.rank != 4:
    print("Ranking error - 2")
    print(dna_4.param_rank_value_for_sort)
if dna_3.rank != 3:
    print("Ranking error - 3")
if dna_2.rank != 2:
    print("Ranking error - 4")
if dna_1.rank != 1:
    print("Ranking error - 5")

print("Test fitness assignemnt")
print("SP = 1")
# Test fitness assignment with SP = 1.0
fitness_assignment(population_set=dna_set, selection_pressure=1.0)
# Check fitness assignment
if dna_1.fitness != 1.0:
    print("Fitness error - 1")
if dna_2.fitness != 1.0:
    print("Fitness error - 2")
if dna_3.fitness != 1.0:
    print("Fitness error - 3")
if dna_4.fitness != 1.0:
    print("Fitness error - 4")
if dna_5.fitness != 1.0:
    print("Fitness error - 5")

print("SP = 2")
# Test fitness assignment with SP = 2.0
fitness_assignment(population_set=dna_set, selection_pressure=2.0)
# Check fitness assignment
if dna_5.fitness != 2.0:
    print(dna_1.fitness)
    print("Fitness error - 1")
if dna_4.fitness != 1.5:
    print("Fitness error - 2")
if dna_3.fitness != 1.0:
    print("Fitness error - 3")
if dna_2.fitness != 0.5:
    print("Fitness error - 4")
if dna_1.fitness != 0.0:
    print("Fitness error - 5")

print("Test sorting by fitness")
# Test selection list sorting
dna_list = list(dna_set)
dna_list.sort(key=lambda x: x.fitness, reverse=True)
# Check list sorting
if dna_list.pop(0).dna_ID != 5:
    print("List sorting error - 1")
if dna_list.pop(0).dna_ID != 4:
    print("List sorting error - 2")
if dna_list.pop(0).dna_ID != 3:
    print("List sorting error - 3")
if dna_list.pop(0).dna_ID != 2:
    print("List sorting error - 4")
if dna_list.pop(0).dna_ID != 1:
    print("List sorting error - 5")

print("Test reinsertion of new childs")
# Create new DNAs acting as children
new_dna_set = set()
# DNA 6
dna_6 = DNA()
dna_6.dna_ID = 6
dna_6.parent_ID = 0
dna_6.max_acc_reached = 0.96
dna_6.fitness = 0.0
dna_6.rank = 0
dna_6.nbr_of_macs = 15000
new_dna_set.add(dna_6)

# DNA 7
dna_7 = DNA()
dna_7.dna_ID = 7
dna_7.parent_ID = 0
dna_7.max_acc_reached = 0.98
dna_7.fitness = 0.0
dna_7.rank = 0
dna_7.nbr_of_macs = 100000
new_dna_set.add(dna_7)

# DNA 8
dna_8 = DNA()
dna_8.dna_ID = 8
dna_8.parent_ID = 0
dna_8.max_acc_reached = 0.88
dna_8.fitness = 0.0
dna_8.rank = 0
dna_8.nbr_of_macs = 0
new_dna_set.add(dna_8)

# Test the reinsertion
reinsert_and_sort(existing_pop_set=dna_set, new_candidate_set=new_dna_set,
                      pop_size=5, min_acc=0.9, criteria="cc")
# Check the reinsertion
# DNA 2, 3, 4, 6, 7 are satisfy min acc of 0.9 -> 1, 5 and 8 are removed
# sorted by # of MACs: 6, 2, 3, 4, 7 -> 5 good, best 4 are 6, 2, 3 and 4
if dna_6 not in dna_set:
    print("Reinsertion error - DNA 6")
if dna_2 not in dna_set:
    print("Reinsertion error - DNA 2")
if dna_3 not in dna_set:
    print("Reinsertion error - DNA 3")
if dna_4 not in dna_set:
    print("Reinsertion error - DNA 4")
if dna_1 in dna_set:
    print("Reinsertion error - DNA 1")
if dna_5 in dna_set:
    print("Reinsertion error - DNA 5")
if dna_7 not in dna_set:
    print("Reinsertion error - DNA 7")
if dna_8 in dna_set:
    print("Reinsertion error - DNA 8")

# End of test
print("TEST END")

