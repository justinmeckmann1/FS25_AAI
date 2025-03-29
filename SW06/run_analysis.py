import pandas as pd
import os
import argparse

path = "log/"
inputfile  = path + "dna_log.txt"
outputfile = path + "analysis.txt"

def printLog(*args, **kwargs):
    print(*args, **kwargs)
    with open(outputfile,'a+') as file:
        print(*args, **kwargs, file=file)

def analyze_results(acc_limit, nbr_dna):

    nbr_top_acc_elements = 8
    nbr_top_macs_elements = 5
    nbr_top_macs_and_acc = 8

    # Import the data
    # Index: dna_id, parent_dna_id, generation, nbr_total_mutations, max_val_acc, nbr_of_macs, nbr_of_param, nbr_of_epochs
    df_data = pd.read_csv(inputfile, sep=',', header=1)

    # Sort by max_val_acc
    df_1 = df_data.sort_values(by='max_val_acc', ascending=False)
    # Extract and print the first n elements
    df_2 = df_1[['dna_id', 'max_val_acc', 'nbr_of_macs', 'nbr_of_param', 'generation', 'nbr_total_mutations']]
    evo_data = df_2\
        .to_records(index=False).tolist()
    printLog("------------------------------------------------------------------------------")
    printLog("Best networks by accuracy only:")
    i = 0
    for dna_data in evo_data:
        i = i + 1
        printLog("{}: DNA {}: Acc={} with {} MACs and {} parameters, Generation {} with {} mutations"
            .format(i, dna_data[0], dna_data[1], dna_data[2], dna_data[3], dna_data[4], dna_data[5]))
        if i == nbr_dna:
            break
    
    # Sort by nbr_of_macs
    df_2 = df_data.sort_values(by='nbr_of_macs', ascending=True)
    # Extract and print the first n elements
    evo_data = df_2[['dna_id', 'max_val_acc', 'nbr_of_macs', 'nbr_of_param', 'generation', 'nbr_total_mutations']]\
        .to_records(index=False).tolist()
    printLog("------------------------------------------------------------------------------")
    printLog("Best networks by number of MACs only:")
    i = 0
    for dna_data in evo_data:
        i = i + 1
        printLog("DNA {}: Acc={} with {} MACs and {} parameters, Generation {} with {} mutations"
            .format(dna_data[0], dna_data[1], dna_data[2], dna_data[3], dna_data[4], dna_data[5]))
        if i == nbr_dna:
            break

    # Filter by max_val_acc and sort by nbr_of_macs
    df_3 = df_data.loc[df_data['max_val_acc'] >= acc_limit]
    df_4 = df_3.sort_values(by='nbr_of_macs', ascending=True)
    # Extract and print the first n elements
    evo_data = df_4[['dna_id', 'max_val_acc', 'nbr_of_macs', 'nbr_of_param', 'generation', 'nbr_total_mutations']]\
        .to_records(index=False).tolist()
    printLog("------------------------------------------------------------------------------")
    printLog("Best networks by number of MACs with accuracy of min. {}:".format(acc_limit))
    i = 0
    for dna_data in evo_data:
        i = i + 1
        printLog("DNA {}: Acc={} with {} MACs and {} parameters, Generation {} with {} mutations"
            .format(dna_data[0], dna_data[1], dna_data[2], dna_data[3], dna_data[4], dna_data[5]))
        if i == nbr_dna:
            break
    printLog("------------------------------------------------------------------------------")

def main():
    parser = argparse.ArgumentParser(description="Analyze results for a certain # of networks from previous EAS run.")
    parser.add_argument("nbr_dna", type=int, help="Max. # of DNAs to report per category, e.g. 10")
    parser.add_argument("acc_limit", type=float, help="Accuracy limit for best networks with min. # of MACs, e.g. 0.85")
    args = parser.parse_args()

    try:
        os.remove(outputfile)
    except OSError:
        pass

    analyze_results(nbr_dna=args.nbr_dna, acc_limit=args.acc_limit)

if __name__ == "__main__":
    main()