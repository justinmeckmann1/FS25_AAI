from dna.DNA import DNA
from log.Arch_log import ArchLog
import config as cfg
import math


class SearchSpaceConstrains(object):
    def __init__(self, restart=False):
        self.max_mac = 0
        self.max_param = 0
        self.time = None
        self.acc_limit_init = 0
        self.acc_step_length = 0
        self.acc_step_height = 0
        self.acc_limit_init_halfM = 0
        self.acc_step_length_halfM = 0
        self.acc_step_height_halfM = 0
        self.static_arch_log_ref = ArchLog(restart=restart)

    # Function to calculate the allowed amount of MAC for a given time limit
    # Input:
    #   throughput: throughput in GMAC/s (Giga MACs per second)
    #   time:       allowed time slot in seconds (dominates over framerate)
    #   frame rate: allowed frame rate in fps
    def calc_max_allowed_mac(self, throughput, time=None, framerate=None):
        # Check if any input is given
        if time is None and framerate is None:
            raise ValueError('To calculate the maximum of allowed MACs give a time or frame rate')
        self.throughput = throughput
        # Proceed with calculating the time slot
        if time is None:
            self.framerate = framerate
            self.time = 1.0/framerate
        else:
            self.time = time
        # Calculate the limit
        self.max_mac = int(self.throughput*1000000000*self.time)
        return self.max_mac

    def set_max_allowed_mac(self, new_maximum: int):
        self.max_mac = new_maximum

    def get_max_mac(self):
        return self.max_mac

    def set_max_allowed_param(self, new_maximum: int):
        self.max_param = new_maximum

    def get_max_param(self):
        return self.max_param

    def set_acc_constrains(self, initial_limit: float, step_duration: int, step_height: float,
                           initial_limit_half_m: float = 0.0, step_duration_half_m: int = 0,
                           step_height_half_m: float = 0.0):
        self.acc_limit_init = initial_limit
        self.acc_step_length = step_duration
        self.acc_step_height = step_height
        self.acc_limit_init_halfM = initial_limit_half_m
        self.acc_step_length_halfM = step_duration_half_m
        self.acc_step_height_halfM = step_height_half_m

    def get_current_acc_limit(self, round: int):
        # Calculate current minimal accuracy limit
        return self.acc_limit_init + (int(round / self.acc_step_length) * self.acc_step_height)

    def get_current_acc_limit_half_m(self, round: int):
        # Calculate current minimal accuracy limit
        return self.acc_limit_init_halfM + (int(round / self.acc_step_length_halfM) * self.acc_step_height_halfM)

    # Function to calculate the number of parameters in a keras model
    def get_nbr_of_parameters(self, keras_model_ref):
        nbr_param = keras_model_ref.count_params()
        return nbr_param

    def is_mac_allowed(self, mac, debug=False):
        if mac <= self.max_mac:
            # Everything ok
            if debug:
                print('Network MAC is in allowed range. It uses {:3.2%}'.format(mac/self.max_mac))
            return True
        else:
            # Too big
            if debug:
                print('Network MAC is NOT in allowed range. It uses {:4.2%}'.format(mac / self.max_mac))
            return False

    def is_param_allowed(self, param, debug=False):
        if param <= self.max_param:
            # Everything ok
            if debug:
                print('Network parameter count is in allowed range. It uses {:3.2%}'.format(param/self.max_param))
            return True
        else:
            # Too big
            if debug:
                print('Network parameter count is NOT in allowed range. It uses {:4.2%}'.format(param / self.max_param))
            return False

    def check_if_dna_allowed(self, dna: DNA, debug=False):
        # Get reference
        keras_model = dna.keras_model_ref
        # Get numbers
        nbr_param = self.get_nbr_of_parameters(keras_model_ref=keras_model)
        nbr_mac = self.get_nbr_of_mac(keras_model_ref=keras_model, debug=debug)
        if cfg.optim_metric == "macs" or cfg.optim_metric == "macs_and_acc":
            # Check if allowed
            complexity_ok = self.is_mac_allowed(mac=nbr_mac, debug=debug)
        else:
            # Also with compl. metric = accuracy, check if DNA is in search space
            complexity_ok = self.is_mac_allowed(mac=nbr_mac, debug=debug)
        if debug:
            print("DNA has {} parameters".format(nbr_param))

        param_ok = self.is_param_allowed(param=nbr_param, debug=debug)
        if complexity_ok and param_ok:
            # Log data in dna
            dna.setNbrOfParameters(nbrOfParam=nbr_param)
            dna.setNbrOfMacs(nbrOfMacs=nbr_mac)
            # Clean up
            del keras_model, nbr_param, complexity_ok, param_ok
            return True
        else:
            # Clean up
            del keras_model, nbr_param, complexity_ok, param_ok
            if cfg.optim_metric == "macs" or cfg.optim_metric == "macs_and_acc":
                del nbr_mac
            return False

    # Function to calculate the number of MAC in a keras model
    # This function works on a layer basis.
    # Currently supported layer types:
    # - Conv2D
    # - Dense
    # A lot of info from: http://machinethink.net/blog/how-fast-is-my-model/
    def get_nbr_of_mac(self, keras_model_ref, debug=False):
        # Variable definition
        mac_total = 0
        # Go through each layer of the model
        for layer in keras_model_ref.layers:
            # Detect class of layer
            layer_type = layer.__class__.__name__
            # If it is a calculation layer record the number of MAC operations
            if layer_type == 'Conv2D':
                # Get the shapes of the input, output and the kernel
                input = layer.input_shape
                input_depth = input[3]
                output = layer.output_shape
                output_h = output[1]
                output_w = output[2]
                output_depth = output[3]
                kernel = layer.kernel.shape
                kernel_r = kernel[0]
                kernel_s = kernel[1]
                # Calculate the MAC operations
                mac = kernel_r * kernel_s * input_depth * output_h * output_w * output_depth
                if debug:
                    print("Conv2D layer detected - MAC:", mac)
            elif layer_type == 'Dense':
                # Get the shape of the input and the hidden layer
                input_length = layer.input_shape[1]
                nbr_of_hidden_layers = layer.kernel.shape[1]
                mac = input_length * nbr_of_hidden_layers
                if debug:
                    print("Dense layer detected - MAC:", mac)
            else:
                if debug:
                    print("Other layer detected:", layer_type)
                mac = 0
            # Count the number of MAC together
            mac_total += mac

        # Report the total amount of MAC
        if debug:
            print("Total amount of MAC:", mac_total)
        return int(mac_total)

   