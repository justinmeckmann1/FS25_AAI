import config as cfg
import numpy as np
# Implementation of dna list sorting
def sort_dna_list_by_criteria(dna_list: list, criteria):
    if criteria == "macs":
        # Sort list by number of macs
        dna_list.sort(key=lambda x: x.nbr_of_macs, reverse=False)
        # Calculate the Value for each parameter
        n = len(dna_list)
        for dna in dna_list:
            dna.param_rank_value_for_sort = n*cfg.WEIGHT_COMPL_METRIC
            n = n-1
    elif criteria == "acc":
        # Sort list by accuracy
        dna_list.sort(key=lambda x: x.max_acc_reached, reverse=True)  # reverse=True for higher acc being better
        return
    else:
        print("SORTING ERROR: Criteria {} unknown".format(criteria))
        return

    # if cfg.ALG_MODE == cfg.ALG_MODE_FIXED_M:
    #     # Sort list by difference between complexity
    #     dna_list.sort(key=lambda x: x.diff_high_low_compl, reverse=False)
    #     n = len(dna_list)
    #     for dna in dna_list:
    #         dna.param_rank_value_for_sort += n*cfg.WEIGHT_DIFF
    #         n = n-1
    #     dna_list.sort(key=lambda x: x.param_rank_value_for_sort, reverse=True) # reverse=True for higher param rank being better
    #     return
    # else:
        # Sort list also by acc
    dna_list.sort(key=lambda x: x.max_acc_reached, reverse=True)  # reverse=True for higher acc being better
    n = len(dna_list)
    for dna in dna_list:
        dna.param_rank_value_for_sort += n*cfg.WEIGHT_ACC
        n = n-1
    dna_list.sort(key=lambda x: x.param_rank_value_for_sort, reverse=True)
    return

# Gaussian function for 2D search
def gaussian_ranking(acc, mac, sigma_x, sigma_y, center_x, center_y):
    xm = acc - center_x
    ym = mac - center_y
    u = np.square(xm/sigma_x) + np.square(ym/sigma_y)
    return float(np.exp(-1*u/2))