from pygal import XY
import pygal
import pandas as pd
import config as cfg
import argparse

def gen_xy_plot(num_gen, min_acc):

    # Variable definition
    path = "log/"
    datafile = 'dna_log.txt'
    fn_data = path + datafile
    nbr_init_dnas = cfg.pop_size
    nbr_of_dnas_per_round = cfg.nbr_of_parents * cfg.nbr_of_children_per_parent
    
    # Define x/y-axis limits
    plots = [(min_acc, 1.0, 0, cfg.max_mac, 'CNN_evolution')]

    # Import the data and sort them by dna_id
    df_data = pd.read_csv(fn_data, sep=',', header=1).sort_values(by='dna_id')
    # Extract plot subset and convert it to plot list
    evo_data = df_data[['nbr_of_macs', 'max_val_acc']].to_records(index=False).tolist()
    evo_dna_id = df_data[['dna_id']].to_records(index=False).tolist()
    evo_dict_list = list()
    for i in range(len(evo_data)):
        tmp_dict = dict()
        tmp_dict['value'] = evo_data[i]
        id_nbr = evo_dna_id[i][0]
        tmp_dict['label'] = 'DNA {}'.format(id_nbr)
        evo_dict_list.append(tmp_dict)

    print("Total # of DNAs created: {}".format(len(evo_dict_list)))

    for plot in plots:
        # Plotting
        xy_chart = XY(stroke=False, dots_size=4, range=(plot[0], plot[1]), xrange=(plot[2], plot[3]),
                    width=1000, x_title='# of MACs', y_title='Validation Accuracy')
        xy_chart.title = 'Evolution of CNN Architectures'
        #xy_chart.x_labels = ['0', '10\'000\'000', '20\'000\'000', '30\'000\'000','40\'000\'000','50\'000\'000']  
        xy_chart.add('Seed DNA', [evo_dict_list[0]])
        xy_chart.add('Init Pop', evo_dict_list[1:nbr_init_dnas])

        i = 1
        tmp_start_old = nbr_init_dnas
        while True:
            i_end = i+(num_gen-1)
            if i_end > 100:
                i_end = 100
            tmp_name = "Gen {}-{}".format(i, i_end)
            tmp_start = tmp_start_old
            tmp_end = tmp_start + num_gen*nbr_of_dnas_per_round
            tmp_data = evo_dict_list[tmp_start:tmp_end]
            tmp_remove_data_list = list()
            # Search elements out of field
            for element in tmp_data:
                tmp_macs = element['value'][0]
                tmp_acc = element['value'][1]
                if tmp_macs > plot[3]:
                    tmp_remove_data_list.append(element)
                elif tmp_macs < plot[2]:
                    tmp_remove_data_list.append(element)
                elif tmp_acc < plot[0]:
                    tmp_remove_data_list.append(element)
            # Remove elements out of field
            for element in tmp_remove_data_list:
                tmp_data.remove(element)
            # Add remaining elements to chart
            xy_chart.add(tmp_name, tmp_data)
            # Prepare next round
            tmp_start_old = tmp_end
            i = i+num_gen
            # Break if the record is not long enought
            if tmp_end >= len(evo_dict_list):
                break
            # Break if finished
            if i+1 > 101:
                break
        # Save plot
    # xy_chart.render_to_png(path + plot[4]+'.png')
    xy_chart.render_to_file(path + plot[4] + '.svg')

def main():
    parser = argparse.ArgumentParser(description="Plot evolution of CNN within the search space")
    parser.add_argument("num_gen", type=int, help="# of generations to group into the same color class, e.g. 1")
    parser.add_argument("min_acc", type=float, help="Lower Accuracy limit for plotting, e.g. 0.85")
    args = parser.parse_args()
    
    gen_xy_plot(num_gen=args.num_gen, min_acc=args.min_acc)

if __name__ == "__main__":
    main()