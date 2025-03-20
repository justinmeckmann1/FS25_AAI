import numpy as np
import config as cfg

# Gaussian function for 2D search
# sigma_x: [%]
# sigma_y: #kMACs
# center_x: (100) [%]
# center_y: #kMACs
def gaussian_ranking(acc, mac, sigma_x, sigma_y, center_x, center_y):
    xm = acc - center_x
    ym = mac - center_y
    u = np.square(xm/sigma_x) + np.square(ym/sigma_y)
    return float(np.exp(-1*u/2))


# Implementation of dna list sorting
def sort_dna_list_by_criteria(dna_list: list, criteria: str = "macs"):
    if criteria == "macs":
        # Sort list by number of macs
        dna_list.sort(key=lambda x: x.nbr_of_macs, reverse=False)
        return

    elif criteria == "acc":
        # Sort list by accuracy
        dna_list.sort(key=lambda x: x.max_acc_reached, reverse=True)  # reverse=True for higher acc being better
        return

    elif criteria == "macs_and_acc":
        # Calculate rank with gaussian function
        # Store the results in the rank variable (temporarily) where it will be overwritten afterwards with the rank
        for dna_element in dna_list:
            dna_element.rank = gaussian_ranking(acc=(dna_element.max_acc_reached*100),
                                                mac=int(np.floor(dna_element.nbr_of_macs / 1000)),
                                                sigma_x=cfg.sigma_acc, sigma_y=cfg.sigma_mac, center_x=cfg.mu_acc, center_y=cfg.mu_mac)

        # Sort list by macs_and_acc rank
        dna_list.sort(key=lambda x: x.rank, reverse=True)  # reverse=True for higher score being better
        return
    else:
        print("SORTING ERROR: Criteria {} unknown".format(criteria))
        return


# For debugging of the gaussian ranking function
# print(gaussian_ranking(100, 5000, sigma_x=10, sigma_y=50000, center_x=100, center_y=5000))
# print(gaussian_ranking(100, 4000, sigma_x=10, sigma_y=50000, center_x=100, center_y=5000))
# print(gaussian_ranking(100, 6000, sigma_x=10, sigma_y=50000, center_x=100, center_y=5000))
# print(gaussian_ranking(100, 3000, sigma_x=10, sigma_y=50000, center_x=100, center_y=5000))
# print(gaussian_ranking(100, 7000, sigma_x=10, sigma_y=50000, center_x=100, center_y=5000))
# print(gaussian_ranking(100, 5000, sigma_x=10, sigma_y=50000, center_x=100, center_y=5000))
# print(gaussian_ranking(100, 5001, sigma_x=10, sigma_y=50000, center_x=100, center_y=5000))
# print(gaussian_ranking(99, 5000, sigma_x=10, sigma_y=50000, center_x=100, center_y=5000))
