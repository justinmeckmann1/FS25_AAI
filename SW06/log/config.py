# Config file for Evolutionary Optimizer

# General
USECASE = "MNIST"  # MNIST | CIFAR
DEBUG = False

Save_init_population = False
Load_init_population = False    # Works just in main_continue_evo.py

# Set the type of the complexity metric (LUT or MAC)
compl_metric = "macs_and_acc"
allowed_compl_metric = ["macs", "macs_and_acc"]

# Usecase: MNIST
if USECASE == "MNIST":
    architecture = "Keras"
    # Initial population
    ground_dna_is_always_parent = True

    # Evolutionary algorithm
    nbr_of_evo_rounds = 3
    pop_size = 4
    nbr_of_selection_picks = 1
    nbr_of_children_per_parent = 2

    # Training
    use_augmentation = False
    batch_size = 100
    nbr_of_training_epochs = 4
    use_early_stopping = True
    early_stop_min_delta = 0.005
    early_stop_patience = 2
    init_LR = 0.0004
    use_ReduceLR = True
    reduce_LR_factor = 0.1
    reduce_LR_min_delta = 0.01
    reduce_LR_patience = 1
    train_dataset_size = 10000
    val_dataset_size = 1000

    # Search space constrains
    max_mac   =  4*10**6  
    max_param =  1*10**5

    # Accuracy limit
    acc_lim_init = 0.90
    acc_lim_step_time = 1
    acc_lim_step_hight = 0.01

    # Mutations
    # --- Dense layer
    dense_mutations = {'dense_insert_layer',
                       'dense_remove_layer',
                       'dense_inc_nbr_neurons',
                       'dense_dec_nbr_neurons',
                       'dense_flip_useDropout',
                       'dense_change_dropout_rate'
                       }
    dense_neuron_numbers = list(range(20, 501, 10))  # Start: 30, Stop: 500, Step: 10
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

# Usecase: CIFAR
if USECASE == "CIFAR":
    architecture = "Kaggle"
    # Initial population
    ground_dna_is_always_parent = True

    # Evolutionary algorithm
    nbr_of_evo_rounds = 8
    pop_size = 6 # 50
    nbr_of_selection_picks = 3 #13
    nbr_of_children_per_parent = 1

    # Training
    use_augmentation = False
    batch_size = 100
    nbr_of_training_epochs = 2 # 30
    use_early_stopping = True
    early_stop_min_delta = 0.005
    early_stop_patience = 3
    use_ReduceLR = True
    init_LR = 0.004
    reduce_LR_factor = 0.5
    reduce_LR_min_delta = 0.01
    reduce_LR_patience = 1
    train_dataset_size = 20000 # max: 
    val_dataset_size = 2000 # max: 

    # Search space constrains
    max_mac =    50000000  # 50*E6
    max_param =   2000000  # 2*E6

    # Accuracy limit
    acc_lim_init = 0.66 
    acc_lim_step_time = 2
    acc_lim_step_hight = 0.01

    # Mutations
    # --- Dense layer
    dense_mutations = {'dense_insert_layer',
                       'dense_remove_layer',
                       'dense_inc_nbr_neurons',
                       'dense_dec_nbr_neurons',
                       'dense_flip_useDropout',
                       'dense_change_dropout_rate'
                       }
    dense_neuron_numbers = list(range(20, 501, 10))  # Start: 30, Stop: 500, Step: 10
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