from pathlib import Path
import numpy as np
import onnxruntime as ort
import tensorrt
import os

from conversion import convert_onnx_fp32_to_fp16, convert_onnx_to_trt_via_trt_python_api, convert_onnx_to_trt_via_trtexec_cli, quantize_onnx_to_trt_via_trt_python_api
from quantization_data_loader import QuantizationDataLoader 

base_path = Path("./")
export_base_path = base_path.joinpath("engines")
model_base_path = base_path.joinpath("models")

model_names = os.listdir(model_base_path.as_posix())

print('####~~~~'*2, " Conversion of ONNX Models to GPU Engines")

for model_name in model_names[0:]:
    print('####~~~~'*1, "Model being converted:", model_name)
    full_model_path = model_base_path.joinpath(model_name)
      
    export_fp32_path = export_base_path.joinpath(model_name.replace(".onnx","_pyapi_fp_32.engine"))
    export_fp16_path = export_base_path.joinpath(model_name.replace(".onnx","_pyapi_fp_16.engine"))
    convert_onnx_to_trt_via_trt_python_api(full_model_path,export_fp32_path,float16=False)
    convert_onnx_to_trt_via_trt_python_api(full_model_path,export_fp16_path,float16=True)

    # not tested on this, first requires data.
    # export_int8_path = export_base_path.joinpath(model_name.replace(".onnx","_cli_int_8.engine"))
     
    # image_path = Path(path_to_around_1000_quantization_images).joinpath("images")
    # data_loader = QuantizationDataLoader(8,
    #                                     image_path,
    #                                     32,
    #                                     32)
      
    # quantize_onnx_to_trt_via_trt_python_api(full_model_path,export_int8_path,int8=True,calibrator="minmax",calibration_stream=data_loader)
    

