from ea.ranking import ranking_macs, ranking_acc, ranking_luts, ranking_luts_and_difference_by_complexity, ranking_by_criteria, ranking_cc
from ea.fitness import fitness_assignment
from ea.elitism import elitism
from log.DNA_log import add_dna_set_to_log
from ea.selection import selection
from ea.mutation_on_list import mutate_dna_list
from training.train_population import train_population
from ea.reinsertion import reinsert_sort_by_macs, reinsert_sort_by_acc, reinsert_sort_by_luts, reinsert_and_sort
from keras.backend import clear_session
import psutil
import os
import config as cfg


def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem


def evolution(population_set, usecase, ssc, log, nbr_of_evo_rounds, debug=False, restart=False, runs_passed=0):
    # Logging start
    population_size = len(population_set)
    log.addToLog("Starting evolution process for {} rounds with a population size of {} candidates."
                 .format(nbr_of_evo_rounds, population_size))

    # Calculate evolution start round
    if restart:
        i = runs_passed
    else:
        i = 0

    # Evolution
    for evo_round in range(i, nbr_of_evo_rounds+i):
        log.addToLog("Evolution round {} started.".format(evo_round+1))

        # Ranking according to config
        if cfg.optim_metric in cfg.allowed_optim_metric:
            r = ranking_by_criteria(population_set= population_set, criteria=cfg.optim_metric)
        else:
            print("No valid compl_metric specified: {} stop executing".format(cfg.optim_metric))
            exit()

        # if cfg.compl_metric == "luts":
        #     if cfg.ALG_MODE == cfg.ALG_MODE_FIXED_M:
        #         r = ranking_luts_and_difference_by_complexity(population_set=population_set)
        #     elif cfg.ALG_MODE == cfg.ALG_MODE_OPTIMUM_M:
        #         r = ranking_luts(population_set=population_set)
        #
        # elif cfg.compl_metric == "cc":
        #     r = ranking_cc(population_set= population_set)
        #
        # elif cfg.compl_metric == "macs":
        #     r = ranking_macs(population_set=population_set)
        #
        # elif cfg.compl_metric == "acc":
        #     r = ranking_acc(population_set=population_set)
        # else:
        #     print("No valid compl_metric specified: {} stop executing".format(cfg.compl_metric))
        #     exit()

        #  r = ranking_macs(population_set=population_set)
        #  r = ranking_luts(population_set=population_set)

        # Set selection pressure
        if nbr_of_evo_rounds > 1:
            SP = float(1+evo_round/(nbr_of_evo_rounds-1))
        else:
            SP = float(1)
        if debug:
            print("Selection pressure is: {}".format(SP))

        # Fitness assignment
        f = fitness_assignment(population_set=population_set, selection_pressure=SP, debug=debug)
        log.addRankedPopulationWithFitness(round_nbr=evo_round, pop_set=population_set)
        # Selection and creation of children from elites
        # elite_set = set()
        # e = elitism(population_set=population_set, elite_set=elite_set, ssc=ssc, nbr_of_elites=1, debug=debug)

        # Selection of parents for mutation
        parent_list = selection(population_set=population_set,
                                candidates_to_pick=cfg.nbr_of_parents, debug=debug)

        # Before using Keras and TF again, clear it to reduce memory consumption
        clear_session()

        # Create a set to hold the new children DNA objects
        children_set = set()
        # Mutation of the survivors
        m = mutate_dna_list(parent_dna_list=parent_list, child_set=children_set, ssc=ssc,
                            nbr_of_children_per_parent=cfg.nbr_of_children_per_parent,
                            round_nbr=evo_round, debug=debug)

        # Create new population set
        new_candidate_set = children_set  # children_set.union(elite_set)

        # Evaluation of the new candidates (train them)
        train_population(population_set=new_candidate_set, usecase=usecase, is_initial_population=False, debug=debug)


        # Calculate and set the difference between the test result of high and low complexity
        # calc_and_set_diff_complexity(population_set=new_candidate_set)

        # Log the dna into the DNA logfile
        # add_dna_set_to_log(population_set=new_candidate_set)

        # Reinsertion
        # for this calculate current minimal accuracy
        acc_limit = ssc.get_current_acc_limit(round=evo_round)
        print("~~~~~~######"*9)
        log.addToLog("Acc Limit = {} in Evolution round {}".format(acc_limit, evo_round+1))
        print("~~~~~~######"*9)

        # if cfg.compl_metric == "luts":
        #     if cfg.ALG_MODE == cfg.ALG_MODE_FIXED_M:
        #         acc_limit_halfM = ssc.get_current_acc_limit_half_m(round=evo_round)
        #         re = reinsert_sort_by_luts(existing_pop_set=population_set, new_candidate_set=new_candidate_set,
        #                                    pop_size=population_size, min_acc=acc_limit, min_acc_halfM=acc_limit_halfM)
        #         del acc_limit_halfM
        #     elif cfg.ALG_MODE == cfg.ALG_MODE_OPTIMUM_M:
        #         re = reinsert_sort_by_luts(existing_pop_set=population_set, new_candidate_set=new_candidate_set,
        #                                    pop_size=population_size, min_acc=acc_limit)
        # elif cfg.compl_metric == "macs":
        #     re = reinsert_sort_by_macs(existing_pop_set=population_set, new_candidate_set=new_candidate_set,
        #                                pop_size=population_size, min_acc=acc_limit)

        re = reinsert_and_sort(existing_pop_set=population_set, new_candidate_set=new_candidate_set,
                               pop_size=population_size, min_acc=acc_limit,
                               criteria=cfg.optim_metric)
        # if cfg.ALG_MODE == cfg.ALG_MODE_FIXED_M:
        #     acc_limit_halfM = ssc.get_current_acc_limit_half_m(round=evo_round)
        #     re = reinsert_and_sort(existing_pop_set=population_set, new_candidate_set=new_candidate_set,
        #                            pop_size=population_size, min_acc=acc_limit, min_acc_halfM=acc_limit_halfM,
        #                            criteria=cfg.compl_metric)
        #     del acc_limit_halfM
        # elif cfg.ALG_MODE == cfg.ALG_MODE_OPTIMUM_M:
        #     re = reinsert_and_sort(existing_pop_set=population_set, new_candidate_set=new_candidate_set,
        #                                pop_size=population_size, min_acc=acc_limit,
        #                                criteria=cfg.compl_metric)

        # re = reinsert_sort_by_acc(existing_pop_set=population_set, new_candidate_set=new_candidate_set,
        #                          pop_size=population_size, min_acc=acc_limit, min_acc_halfM=acc_limit_halfM)

        # Clean up
        del parent_list, children_set, new_candidate_set, acc_limit, r, f, m, re  # ,e, elite_set

        # Finish round
        log.addToLog("Evolution round {} finished. Population size: {}".format(evo_round+1, len(population_set)))
        log.addPopulationSet(round_nbr=evo_round+1, pop_set=population_set)
        log.addTimeToLog()

        # Debug memory leaks
        #log.addToLog("--- DEBUG --- : Memory consumption after round {}: {}".format(evo_round+1, memory_usage_psutil()))
