from pathlib import Path
import numpy as np
import onnxruntime as ort
import tensorrt
import os

from inference import single_engine_cli_inference, single_engine_inference_python_api, single_onnx_inference

base_path = Path("./")
engines_base_path = base_path.joinpath("engines")

image_path = Path("images/best_model_dna_0_input.npy")
expected_result_path = Path("images/best_model_dna_0_output.npy")

model_base_path = base_path.joinpath("models")
model_names = os.listdir(model_base_path.as_posix())

image_base_path = base_path.joinpath(image_path)
images = np.load(image_base_path) 

expected_result_base_path = base_path.joinpath(expected_result_path)
expected_result = np.load(expected_result_base_path) 

print('####~~~~'*2, "Inference with TRT Engine on GPU ")
engine_inference_model_names = os.listdir(engines_base_path.as_posix())
engine_inference_model_names = [file for file in engine_inference_model_names if file.endswith(".engine")]

inference_output_errors = []

for model_name in engine_inference_model_names:
    full_model_path = engines_base_path.joinpath(model_name)
    try:
        inference_output_errors.append(single_engine_inference_python_api(full_model_path, images, expected_result))
        #single_engine_cli_inference(full_model_path, images)
    except Exception as e:
        print(e)

m = 0
for out_error in inference_output_errors:
    print('####~~~~'*1, "Max. Inference error of : ", engine_inference_model_names[m], " = ", np.max(np.abs(out_error)))
    m += 1   

print('####~~~~'*2, "Inference with ONNX Model on CPU")
onnx_inference_model_names = os.listdir(model_base_path.as_posix())

inference_output_errors = []

for model_name in onnx_inference_model_names:
    full_model_path = model_base_path.joinpath(model_name)
    try:
        inference_output_errors.append(single_onnx_inference(full_model_path, images, expected_result))
    except Exception as e:
        print(e)

m = 0
for out_error in inference_output_errors:
    print('####~~~~'*1, "Max. Inference error of : ", onnx_inference_model_names[m], " = ", np.max(np.abs(out_error)))
    m += 1   



