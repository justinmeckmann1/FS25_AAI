from dask import callbacks
import timeit

from dna.DNA import DNA
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from log.DNA_log import add_dna_to_log
from log.Logging import MainLog
from log.DNA_weights_log import add_dna_weights_to_log
import config as cfg

def estimate_runtime(training_time):
    nbr_of_trainings = cfg.pop_size + (cfg.nbr_of_parents * cfg.nbr_of_children_per_parent * cfg.nbr_of_evo_rounds)
    runtime = nbr_of_trainings * training_time
    print("-"*70)
    print("Estimated runtime for full EA based on this training: {:.0f} to {:.0f} mins." .format(0.8*runtime/60, 1.2* runtime/60))
    print("-"*70)
    return runtime
    


# Function to train a DNA model
def train_dna(dna, usecase, is_initial_population):
    assert isinstance(dna, DNA)
    # Variable definitions
    fn_csv_log = 'log/training_logs/training_log_dna_' + str(dna.dna_ID) + '.txt'
    fn_model_log = 'log/training_models/best_model_dna_' + str(dna.dna_ID) + '.h5'

    # Compile model
    dna.keras_model_ref.compile(loss='categorical_crossentropy',
                                optimizer=Adam(learning_rate=dna.learning_rate),
                                metrics=['categorical_accuracy'])

    if cfg.DEBUG:
        # Prepare callbacks
        callback = [
            CSVLogger(filename=fn_csv_log),
            ModelCheckpoint(filepath=fn_model_log, monitor='val_categorical_accuracy', verbose=1,
                            save_best_only=True, save_weights_only=False)
            ]
        if cfg.use_early_stopping:
            callback.append(EarlyStopping(monitor='val_categorical_accuracy', min_delta=cfg.early_stop_min_delta,
                                          patience=cfg.early_stop_patience, verbose=1))
        if cfg.use_ReduceLR:
            callback.append(ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=cfg.reduce_LR_factor,
                                              min_delta=cfg.reduce_LR_min_delta, patience=cfg.reduce_LR_patience,
                                              verbose=1))
    else:
        # Prepare callbacks
        callback = [
            CSVLogger(filename=fn_csv_log),
            ModelCheckpoint(filepath=fn_model_log, monitor='val_categorical_accuracy', verbose=0,
                            save_best_only=True, save_weights_only=False)
            ]
        if cfg.use_early_stopping:
            callback.append(EarlyStopping(monitor='val_categorical_accuracy', min_delta=cfg.early_stop_min_delta,
                                          patience=cfg.early_stop_patience, verbose=0))
        if cfg.use_ReduceLR:
            callback.append(ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=cfg.reduce_LR_factor,
                                              min_delta=cfg.reduce_LR_min_delta, patience=cfg.reduce_LR_patience,
                                              verbose=0))

    # Train the model 
    print("Training DNA number:", str(dna.dna_ID))
    if cfg.DEBUG:
        dna.keras_model_ref.fit(usecase.training_data, 
                                validation_data=usecase.validation_data,
                                epochs=cfg.nbr_of_training_epochs, 
                                verbose=2,
                                callbacks=callback)
    else:
        dna.keras_model_ref.fit(usecase.training_data, 
                                validation_data=usecase.validation_data,
                                epochs=cfg.nbr_of_training_epochs, 
                                verbose=2,
                                callbacks=callback)


    # Log the dna into the DNA logfile
    add_dna_to_log(dna=dna)

    # if configured in the config.py > Save the CNN (incl. weights) to file
    if is_initial_population and cfg.Save_init_population:
        add_dna_weights_to_log(dna=dna)
    del callback, usecase


def train_population(population_set: set, usecase, is_initial_population, debug=False):
    tot_runtime = 0
    # Loop over all DNAs in the population
    for dna in population_set:
        # Get runtimes for initial population
        if is_initial_population: 
            start_time = timeit.default_timer()
        # Compile and train the model
        train_dna(dna=dna, usecase=usecase, is_initial_population=is_initial_population)
        # Calculate runtime for initial population
        if is_initial_population: 
            end_time = timeit.default_timer()
            tot_runtime += (estimate_runtime(end_time - start_time))
    if is_initial_population:
        return tot_runtime / cfg.pop_size
    return 0

