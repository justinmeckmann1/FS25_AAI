from log.Logging import MainLog
from log.DNA_log import add_dna_set_to_log
from ssc.search_space_constraints import SearchSpaceConstrains
from MNIST.Usecase_MNIST import MNISTUseCase
from CIFAR.Usecase_CIFAR import CIFARUseCase
from tensorflow.python.client import device_lib
from dna.DNA import DNA
from dna.gen_init_pop import generateInitialPopulation
from training.train_population import train_population
from keras.backend import clear_session
from ea.evolution import evolution
import config as cfg
import tensorflow as tf
from tensorflow.keras import mixed_precision
import os
import time


if __name__ == '__main__':
    try: 
        print("Running EA... Press Ctrl+C to exit.")
        # Tensorflow - check available hardware
        clear_session()
        print("Device list for tensorflow:")
        print(device_lib.list_local_devices())
        
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'

        # This can speed-up training if a GPU is used - uncomment to use
        # policy = mixed_precision.Policy('mixed_float16')
        # mixed_precision.set_global_policy(policy)

        # Main logging initialization
        log = MainLog()
        log.printTimeToConsole()

        # Generate search space constrains
        ssc = SearchSpaceConstrains()
        # Check and set MAC limit
        ssc.set_max_allowed_mac(new_maximum=cfg.max_mac)

        # Set parameter limit
        ssc.set_max_allowed_param(new_maximum=cfg.max_param)
        log.addToLog("Maximum MACs allowed: {} - Maximum parameters allowed: {}"
                    .format(ssc.get_max_mac(), ssc.get_max_param()))

        ssc.set_acc_constrains(initial_limit=cfg.acc_lim_init, step_duration=cfg.acc_lim_step_time,
                            step_height=cfg.acc_lim_step_hight)

        # switch processing order for all Conv/Pool/Norm/Reshape layers
        # from default = 'channel_last' to 'channel first' in order to 
        # ensure concistency with backend tools
        # not working because: Default MaxPoolingOp only supports NHWC on device type CPU #################
        #tf.keras.backend.set_image_data_format('channels_first')

        # Usecase data preparation
        if cfg.USECASE == "MNIST":
            # Usecase data preparation
            usecase = MNISTUseCase(debug=cfg.DEBUG)
            log.addToLog("Usecase MNIST - data prepared")
            # Load basic architecture
            basicArchitecture = usecase.get_reference_model(cfg.architecture)
            log.addToLog("Usecase MNIST - Architecture {} loaded".format(cfg.architecture))
        elif cfg.USECASE == "CIFAR": 
            usecase = CIFARUseCase(debug=cfg.DEBUG)
            log.addToLog("Usecase CIFAR - data prepared")
            basicArchitecture = usecase.get_reference_model(cfg.architecture)
            log.addToLog("Usecase CIFAR - Architecture {} loaded".format(cfg.architecture))
        else:
            print("ERROR: Uscase {} unknown".format(cfg.USECASE))
            exit()

        # Create and build basic DNA
        basicDNA = DNA()
        basicDNA.setInputShape(shape=usecase.input_shape)
        basicDNA.setArchitecture(architecture=basicArchitecture)
        basicDNA.setDNA_ID()
        log.addToLog(">>>>>>>> Basic DNA created")

        # Generate initial population
        dna_set = generateInitialPopulation(ground_dna=basicDNA, ssc=ssc, population_size=cfg.pop_size, debug=cfg.DEBUG)
        log.addToLog(">>>>>>>> Initial population created. Population count: {}. Starting initial training.".format(len(dna_set)))

        # Train initial population
        avg_runtime = train_population(population_set=dna_set, usecase=usecase, is_initial_population=True, debug=cfg.DEBUG)
        log.addToLog(">>>>>>>> Initial population trained.")
        log.addToLog("-"*55)
        log.addToLog("Estimated runtime for full EA is {:.0f} to {:.0f} mins.".format(0.8*avg_runtime/60, 1.2*avg_runtime/60))
        log.addToLog("-"*55)
        log.printTimeToConsole()
        log.addPopulationSet(round_nbr=0, pop_set=dna_set)

        # Clean up
        del basicArchitecture, basicDNA
        clear_session()

        # Run evolution
        evolution(population_set=dna_set, usecase=usecase, ssc=ssc, log=log,
                nbr_of_evo_rounds=cfg.nbr_of_evo_rounds, debug=cfg.DEBUG)

        print(">>>>>>>> End of EAS")
    except KeyboardInterrupt: 
        print("\Exiting gracefully...")