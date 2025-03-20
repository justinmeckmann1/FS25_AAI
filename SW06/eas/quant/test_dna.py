import config as cfg


# Function to test the DNA
def test_dna(dna, usecase, debug=False):
    if debug:
        print("Test the quantized DNA number:", str(dna.dna_ID))
    # Test the quantized dna
    accuracy = dna.keras_model_ref.evaluate(x=usecase.validation_data,
                                            y=usecase.validation_label,
                                            batch_size=cfg.batch_size,
                                            verbose=0)
    return accuracy[1]

