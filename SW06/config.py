# Config file for Evolutionary Architecture Search (EAS)
#######################################################################################

# Select Usecase
USECASE = "CIFAR"  # MNIST | CIFAR

# Select optimization metric to be used for EAS
allowed_optim_metric = ["macs", "acc", "macs_and_acc"]
optim_metric = "macs_and_acc"  # macs | acc | macs_and_acc
# Power-user parameters (do not change)
DEBUG = False
Save_init_population = False
Load_init_population = False  # Works just in main_continue_evo.py


######################################################################################
##### Usecase: CIFAR #################################################################
######################################################################################
if USECASE == "CIFAR":
    architecture = "Kaggle"
    
    ##### Initial population generation ##############################################
    # Parent DNA selection
    ground_dna_is_always_parent = False # True

    ##### Search space constraints ###################################################
    # Complexity constraints (checked with each mutation)
    max_mac            = 40*10**6                                                       # <---- ToDo
    max_param          = 1*10**6                                                       # <---- ToDo
    acc_lim_init       = 0.70                                                         # <---- ToDo
    acc_lim_step_time  = 1      # default 1
    acc_lim_step_hight = 0.01   # default 0.01

    ##### Evolution parameters #######################################################
    pop_size                   = 10 #10
    nbr_of_evo_rounds          = 30 #4
    nbr_of_parents             = 4 
    nbr_of_children_per_parent = 2

    ##### Ranking parameters #########################################################
    # 2D-Gaussian function parameters (used with metric "macs_and_acc")
    # The line `nbr_of_evo_rounds = 12 #4` is setting the variable `nbr_of_evo_rounds` to the value
    # 12, and the comment `#4` is providing additional information or context about the value. In this
    # case, it seems like the comment is indicating that the value 4 was previously considered or used
    # for this variable. However, the current value being set is 12. The comment is there to help
    # understand the history or reasoning behind the choice of the value.
    mu_acc    = 80         # expected value accuracy [%]                            # <---- ToDo
    sigma_acc = 10          # standard deviation accuracy [%]                        # <---- ToDo
    mu_mac    = 12*10**6  # expected value # of MACs                               # <---- ToDo
    sigma_mac =20*10**6  # standard deviation # of MACs                           # <---- ToDo

    ##### Training parameters ########################################################
    # General parameters
    nbr_of_training_epochs = 12
    batch_size             = 200
    train_dataset_size     = 30000
    val_dataset_size       = 3000
    use_augmentation       = False
    # Early stopping parameters
    use_early_stopping     = True
    early_stop_min_delta   = 0.005
    early_stop_patience    = 4 # 3
    # Dynamic learning rate parameters
    use_ReduceLR           = True
    init_LR                = 0.004
    reduce_LR_factor       = 0.5
    reduce_LR_min_delta    = 0.01
    reduce_LR_patience     = 1

    # Mutations
    # --- Dense layer
    dense_mutations = {'dense_insert_layer',
                       'dense_remove_layer',
                       'dense_inc_nbr_neurons',
                       'dense_dec_nbr_neurons',
                       'dense_flip_useDropout',
                       'dense_change_dropout_rate'
                       }
    dense_neuron_numbers = list(range(20, 501, 10))  # Start: 20, Stop: 500, Step: 10
    dense_activations = ['relu']
    dense_dropout_rates = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]

    # --- Convolution layer
    conv_mutations = {'conv_insert_layer',
                      'conv_remove_layer',
                      'conv_alter_stride',
                      'conv_alter_nbr_filters',
                      'conv_alter_filter_size',
                      'conv_change_padding',
                      'conv_flip_useBN',
                      'conv_flip_useDropout',
                      'conv_change_dropout_rate'
                      }
    conv_stride_numbers = [1, 2, 4]
    conv_filter_numbers = list(range(1, 129, 8))  # Start: 1, Stop: 128, Step: 8
    conv_kernel_sizes = list(range(3, 8, 2))  # Start: 3, End: 7, Step: 2
    conv_paddings = ['same', 'valid']
    conv_activations = ['relu']
    conv_dropout_rates = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]

    # --- Pooling layer
    pooling_mutations = {'pool_insert_layer',
                         'pool_remove_layer',
                         'pool_alter_stride',
                         'pool_alter_kernel_size',
                         'pool_change_padding'
                         }
    pool_stride_numbers = [1, 2, 4]
    pool_kernel_sizes = list(range(2, 8, 1))  # Start: 2, End: 7, Step: 1
    pool_paddings = ['same', 'valid']

    # --- Dropout layer
    dropout_mutations = {'dropout_add_layer',
                        'dropout_remove_layer',
                        'dropout_alter_rate'
                        }
    drop_dropout_rates = [0.25, 0.5, 0.75]

    # Create a list with all allowed mutations
    allowed_mutations = list(dense_mutations.union(conv_mutations, pooling_mutations, dropout_mutations))