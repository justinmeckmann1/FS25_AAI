from dna.DNA import DNA
from ea.sorting_v2 import sort_dna_list_by_criteria
import config as cfg

# Reinsert the child population into the existing population with respect to the number of macs
def reinsert_sort_by_macs(existing_pop_set: set, new_candidate_set: set, pop_size: int, min_acc, min_acc_half_M = 0.0):
    too_low_acc_set = set()
    # Check if new candidates have reached the minimal required accuracy
    for dna in new_candidate_set:
        assert isinstance(dna, DNA)
        # Check if accuracy is smaller then required
        if dna.max_acc_reached < min_acc:
            # If accuracy is to small, we move the dna to an other set
            too_low_acc_set.add(dna)
            print("Removed DNA {} because of bad acc of {}".format(dna.dna_ID,
                                                                                      dna.max_acc_reached))
        if dna.max_acc_reached_halfM < min_acc_half_M:
            too_low_acc_set.add(dna)
            print("Removed DNA {} because of bad acc with half M complexity of {}".format(dna.dna_ID,
                                                                                     dna.max_acc_reached_halfM))

    # Remove moved elements from original set
    for dna in too_low_acc_set:
        new_candidate_set.discard(dna)

    # Check if the old candidates have reached the minimal required accuracy
    for dna in existing_pop_set:
        assert isinstance(dna, DNA)
        # Check if accuracy is smaller then required
        if dna.max_acc_reached < min_acc:
            # If accuracy is to small, we move the dna to an other set
            too_low_acc_set.add(dna)
            print("Removed DNA {} because of bad acc of {}".format(dna.dna_ID, dna.max_acc_reached))
        if dna.max_acc_reached_halfM < min_acc_half_M:
            too_low_acc_set.add(dna)
            print("Removed DNA {} because of bad acc with half M complexity of {}".format(dna.dna_ID, dna.max_acc_reached_halfM))

    # Remove moved elements from original set
    for dna in too_low_acc_set:
        existing_pop_set.discard(dna)

    # Create a set with the good DNAs from both sets
    good_candidates_set = existing_pop_set.union(new_candidate_set)

    # Check the length of the good set
    back_list = list()
    if len(good_candidates_set) < pop_size:
        # Not enought good candidates. We add candidates with the next best accuracy to fill the set
        # Calculate the amount of missing candidates
        n_missing = pop_size - len(good_candidates_set)
        print("Good set is missing {} DNAs".format(n_missing))
        # Sort the too low acc set by acc
        too_low_acc_list = list(too_low_acc_set)
        too_low_acc_list.sort(key=lambda x: x.max_acc_reached, reverse=False)
        # Take the best candidates and add them back to the good set
        back_list.extend(too_low_acc_list[:n_missing])

    # Create a big list with the old and new DNAs
    pop_list = list(good_candidates_set)
    print("Poplist length: {}".format(len(pop_list)))

    # Sort list by number of macs
    sort_dna_list_by_criteria(dna_list=pop_list, criteria=cfg.optim_metric)

    # Append the bad candidates to fill the list if needed
    pop_list.extend(back_list)

    # Now take the best n individuals to form the new population and
    # store them in place of the existing population
    existing_pop_set.clear()
    existing_pop_set.update(pop_list[:pop_size])

    # Clean up
    del dna, pop_list, back_list


# Reinsert the child population into the existing population with respect to the number of luts
def reinsert_sort_by_luts(existing_pop_set: set, new_candidate_set: set, pop_size: int, min_acc, min_acc_halfM = 0.0):
    too_low_acc_set = set()
    # Check if new candidates have reached the minimal required accuracy
    for dna in new_candidate_set:
        assert isinstance(dna, DNA)
        # Check if accuracy is smaller then required
        if dna.max_acc_reached < min_acc:
            # If accuracy is to small, we move the dna to an other set
            too_low_acc_set.add(dna)
            print("Removed DNA {} because of bad acc of {}".format(dna.dna_ID,
                                                                                      dna.max_acc_reached))
        if dna.max_acc_reached_halfM < min_acc_halfM:
            too_low_acc_set.add(dna)
            print("Removed DNA {} because of bad acc with half M complexity of {}".format(dna.dna_ID,
                                                                                     dna.max_acc_reached_halfM))

    # Remove moved elements from original set
    for dna in too_low_acc_set:
        new_candidate_set.discard(dna)

    # Check if the old candidates have reached the minimal required accuracy
    for dna in existing_pop_set:
        assert isinstance(dna, DNA)
        # Check if accuracy is smaller then required
        if dna.max_acc_reached < min_acc:
            # If accuracy is to small, we move the dna to an other set
            too_low_acc_set.add(dna)
            print("Removed DNA {} because of bad acc of {}".format(dna.dna_ID, dna.max_acc_reached))
        if dna.max_acc_reached_halfM < min_acc_halfM:
            too_low_acc_set.add(dna)
            print("Removed DNA {} because of bad acc with half M complexity of {}".format(dna.dna_ID, dna.max_acc_reached_halfM))

    # Remove moved elements from original set
    for dna in too_low_acc_set:
        existing_pop_set.discard(dna)

    # Create a set with the good DNAs from both sets
    good_candidates_set = existing_pop_set.union(new_candidate_set)

    # Check the length of the good set
    back_list = list()
    if len(good_candidates_set) < pop_size:
        # Not enought good candidates. We add candidates with the next best accuracy to fill the set
        # Calculate the amount of missing candidates
        n_missing = pop_size - len(good_candidates_set)
        print("Good set is missing {} DNAs".format(n_missing))
        # Sort the too low acc set by acc
        too_low_acc_list = list(too_low_acc_set)
        too_low_acc_list.sort(key=lambda x: x.max_acc_reached, reverse=False)
        # Take the best candidates and add them back to the good set
        back_list.extend(too_low_acc_list[:n_missing])

    # Create a big list with the old and new DNAs
    pop_list = list(good_candidates_set)
    print("Poplist length: {}".format(len(pop_list)))

    # Sort list by number of macs
    sort_dna_list_by_criteria(dna_list=pop_list, criteria="luts")

    # Append the bad candidates to fill the list if needed
    pop_list.extend(back_list)

    # Now take the best n individuals to form the new population and
    # store them in place of the existing population
    existing_pop_set.clear()
    existing_pop_set.update(pop_list[:pop_size])

    # Clean up
    del dna, pop_list, back_list


# Reinsert the child population into the existing population with respect to the accuracy
def reinsert_sort_by_acc(existing_pop_set: set, new_candidate_set: set, pop_size: int, min_acc, min_acc_halfM = 0.0):
    too_low_acc_set = set()
    # Check if new candidates have reached the minimal required accuracy
    for dna in new_candidate_set:
        assert isinstance(dna, DNA)
        # Check if accuracy is smaller then required
        if dna.max_acc_reached < min_acc:
            # If accuracy is to small, we move the dna to an other set
            too_low_acc_set.add(dna)
            print("Removed DNA {} because of bad acc of {}".format(dna.dna_ID, dna.max_acc_reached))
        if dna.max_acc_reached_halfM < min_acc_halfM:
            too_low_acc_set.add(dna)
            print("Removed DNA {} because of bad acc with half M complexity of {}".format(dna.dna_ID, dna.max_acc_reached_halfM))

    # Remove moved elements from original set
    for dna in too_low_acc_set:
        new_candidate_set.discard(dna)

    # Check if the old candidates have reached the minimal required accuracy
    for dna in existing_pop_set:
        assert isinstance(dna, DNA)
        # Check if accuracy is smaller then required
        if dna.max_acc_reached < min_acc:
            # If accuracy is to small, we move the dna to an other set
            too_low_acc_set.add(dna)
            print("Removed DNA {} because of bad acc of {}".format(dna.dna_ID, dna.max_acc_reached))
        if dna.max_acc_reached_halfM < min_acc_halfM:
            too_low_acc_set.add(dna)
            print("Removed DNA {} because of bad acc with half M complexity of {}".format(dna.dna_ID, dna.max_acc_reached_halfM))

    # Remove moved elements from original set
    for dna in too_low_acc_set:
        existing_pop_set.discard(dna)

    # Create a set with the good DNAs from both sets
    good_candidates_set = existing_pop_set.union(new_candidate_set)

    # Check the length of the good set
    back_list = list()
    if len(good_candidates_set) < pop_size:
        # Not enought good candidates. We add candidates with the next best accuracy to fill the set
        # Calculate the amount of missing candidates
        n_missing = pop_size - len(good_candidates_set)
        print("Good set is missing {} DNAs".format(n_missing))
        # Sort the too low acc set by acc
        too_low_acc_list = list(too_low_acc_set)
        too_low_acc_list.sort(key=lambda x: x.max_acc_reached, reverse=False)
        # Take the best candidates and add them back to the good set
        back_list.extend(too_low_acc_list[:n_missing])

    # Create a big list with the old and new DNAs
    pop_list = list(good_candidates_set)
    print("Poplist length: {}".format(len(pop_list)))

    # Sort list by number of macs
    sort_dna_list_by_criteria(dna_list=pop_list, criteria="acc")

    # Append the bad candidates to fill the list if needed
    pop_list.extend(back_list)

    # Now take the best n individuals to form the new population and
    # store them in place of the existing population
    existing_pop_set.clear()
    existing_pop_set.update(pop_list[:pop_size])

    # Clean up
    del dna, pop_list, back_list

# Reinsert the child population into the existing population with respect to the accuracy
def reinsert_and_sort(existing_pop_set: set, new_candidate_set: set, pop_size: int, min_acc, min_acc_halfM = 0.0, criteria:str= "acc"):
    too_low_acc_set = set()
    # Check if new candidates have reached the minimal required accuracy
    for dna in new_candidate_set:
        assert isinstance(dna, DNA)
        # Check if accuracy is smaller then required
        if dna.max_acc_reached < min_acc:
            # If accuracy is to small, we move the dna to an other set
            too_low_acc_set.add(dna)
            print("Removed DNA {} because of bad acc of {}".format(dna.dna_ID, dna.max_acc_reached))
        if dna.max_acc_reached_halfM < min_acc_halfM:
            too_low_acc_set.add(dna)
            print("Removed DNA {} because of bad acc with half M complexity of {}".format(dna.dna_ID, dna.max_acc_reached_halfM))

    # Remove moved elements from original set
    for dna in too_low_acc_set:
        new_candidate_set.discard(dna)

    # Check if the old candidates have reached the minimal required accuracy
    for dna in existing_pop_set:
        assert isinstance(dna, DNA)
        # Check if accuracy is smaller then required
        if dna.max_acc_reached < min_acc:
            # If accuracy is to small, we move the dna to an other set
            too_low_acc_set.add(dna)
            print("Removed DNA {} because of bad acc of {}".format(dna.dna_ID, dna.max_acc_reached))
        if dna.max_acc_reached_halfM < min_acc_halfM:
            too_low_acc_set.add(dna)
            print("Removed DNA {} because of bad acc with half M complexity of {}".format(dna.dna_ID, dna.max_acc_reached_halfM))

    # Remove moved elements from original set
    for dna in too_low_acc_set:
        existing_pop_set.discard(dna)

    # Create a set with the good DNAs from both sets
    good_candidates_set = existing_pop_set.union(new_candidate_set)

    # Check the length of the good set
    back_list = list()
    if len(good_candidates_set) < pop_size:
        # Not enought good candidates. We add candidates with the next best accuracy to fill the set
        # Calculate the amount of missing candidates
        n_missing = pop_size - len(good_candidates_set)
        print("Good set is missing {} DNAs".format(n_missing))
        # Sort the too low acc set by acc
        too_low_acc_list = list(too_low_acc_set)
        too_low_acc_list.sort(key=lambda x: x.max_acc_reached, reverse=False)
        # Take the best candidates and add them back to the good set
        back_list.extend(too_low_acc_list[:n_missing])

    # Create a big list with the old and new DNAs
    pop_list = list(good_candidates_set)
    print("Poplist length: {}".format(len(pop_list)))

    # Sort list by number of macs
    sort_dna_list_by_criteria(dna_list=pop_list, criteria=criteria)

    # Append the bad candidates to fill the list if needed
    pop_list.extend(back_list)

    # Now take the best n individuals to form the new population and
    # store them in place of the existing population
    existing_pop_set.clear()
    existing_pop_set.update(pop_list[:pop_size])

    # Clean up
    del dna, pop_list, back_list