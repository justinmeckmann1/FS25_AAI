import matplotlib.pyplot as plt
import numpy as np
import config as cfg
import argparse

def gen_2d_plot(mac_0,acc_0,mac_t,acc_t):

    # Get search space constraints from configuration
    ssc_mac = cfg.max_mac
    ssc_acc = cfg.acc_lim_init*100

    # Get 2D Gaussian function parameters from configuration
    mu_mac = cfg.mu_mac
    si_mac = cfg.sigma_mac
    mu_acc = cfg.mu_acc
    si_acc = cfg.sigma_acc

    # Generating x,y grid points
    x_mac = np.linspace(0, ssc_mac, 1000)
    y_acc = np.linspace(ssc_acc, 100, 1000)

    X, Y = np.meshgrid(x_mac, y_acc)

    # 2D Gaussian ranking function
    u = np.square((X-mu_mac)/si_mac) + np.square((Y-mu_acc)/si_acc)  
    Z = np.exp(-1*(u)/2)

    # Create filled contour plot
    plt.contourf(X, Y, Z, levels = 10)
    #plt.colorbar()

    # Add DNA 0 position
    dna_0, = plt.plot(mac_0, acc_0, 'ro')
    dna_0.set_label('DNA 0')
    plt.legend(loc='lower right')

    # Add DNA Target position
    dna_t, = plt.plot(mac_t, acc_t, 'go')
    dna_t.set_label('DNA Target')
    plt.legend(loc='lower right')

    # # Adding labels and title
    plt.xlabel('# of MACs')
    plt.ylabel('Accuracy [%]')
    plt.title('2D Gaussian ranking function')

    # Display the plot
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot 2D Gaussian ranking function")
    parser.add_argument("mac_0", type=int, help="# of MACs of DNA 0, e.g. 20*10**6")
    parser.add_argument("acc_0", type=float, help="Accuracy of DNA 0 [%], e.g. 84.5")
    parser.add_argument("mac_t", type=int, help="# of MACs of Traget DNA, e.g. 2*10**6")
    parser.add_argument("acc_t", type=float, help="Accuracy of Target DNA [%], e.g. 96.0")
    args = parser.parse_args()
    
    gen_2d_plot(mac_0=args.mac_0, acc_0=args.acc_0,mac_t=args.mac_t, acc_t=args.acc_t)

if __name__ == "__main__":
    main()