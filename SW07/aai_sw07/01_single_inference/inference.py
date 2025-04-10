
import subprocess
import numpy as np
import tensorrt as trt
import onnxruntime as ort
from pathlib import Path
import pycuda.driver as cuda
import pycuda.autoinit


def single_onnx_inference(model_path:Path, images:np.array, expected_result:np.array):
    
    #onnx_inference_providers = ort.get_available_providers()
    onnx_inference_providers = ['CPUExecutionProvider']
    
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    inference_outputs = []
    for provider in onnx_inference_providers:
        #print("Inference {} using {}".format(model_path.name,provider))
        ort_sess = ort.InferenceSession(model_path.as_posix(), options, providers=[provider])
        input_shape = ort_sess.get_inputs()[0].shape

        output_layer_name = ort_sess.get_outputs()[0].name
        input_layer_name = ort_sess.get_inputs()[0].name
        
        output = ort_sess.run([output_layer_name], {input_layer_name: images})
        inference_outputs.append(expected_result - output)

    return inference_outputs

def single_engine_cli_inference(model_path:Path, images:np.array):
 
    command = [
        "/usr/src/tensorrt/bin/trtexec", 
        f"--loadEngine={model_path.as_posix()}",
        "--avgRuns=100"
    ]
    
    # Run the command
    subprocess.run(command, check=True)
    
    
def single_engine_inference_python_api(model_path:Path, images:np.array, expected_result:np.array):
    
    #print("Inference {} using TensorRT".format(model_path.name))
    
    logger = trt.Logger(trt.Logger.ERROR)
    with open(model_path.as_posix(), "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    inputs = []
    outputs = []
    allocations = []

    #print(">>>>>>>>>>>>>>>>>>>>>> 1", context)

    for i in range(engine.num_bindings):
        is_input = False
        if engine.binding_is_input(i):
            is_input = True
        name = engine.get_binding_name(i)
        dtype = engine.get_binding_dtype(i)
        shape = engine.get_binding_shape(i)

        #print(">>>>>>>>>>>>>>>>>>>>>> 2: i = ", i)

        if is_input:
            batch_size = shape[0]
            
        if dtype == trt.float32:
            defined_type = np.float32()
            
        elif dtype in [trt.DataType.HALF, trt.float16]:
            defined_type = np.float16()
        elif dtype in [trt.DataType.INT32, trt.int32]:
            defined_type = np.int32()
        elif dtype in [trt.DataType.INT8, trt.int8]:
            defined_type = np.int8()
        elif dtype in [trt.DataType.BOOL, trt.bool]:
            defined_type = np.bool_()
        elif dtype in [trt.DataType.UINT8, trt.uint8]:
            defined_type = np.uint8()
        elif dtype in [trt.DataType.FP8, trt.fp8]:
            # Assuming FP8 corresponds to numpy float8
            defined_type = np.dtype('float8')
        elif dtype in [trt.DataType.BF16, trt.bfloat16]:
            defined_type = np.dtype('bfloat16')
        elif dtype in [trt.DataType.INT64, trt.int64]:
            defined_type = np.int64()
        else:
            raise ValueError("Unsupported dtype")
        size = defined_type.itemsize
        #size = np.dtype(trt.nptype(dtype)).itemsize

        for s in shape:
            size *= s

        allocation = cuda.mem_alloc(size)
        binding = {
            'index': i,
            'name': name,
            'dtype': defined_type,
            'shape': list(shape),
            'allocation': allocation,
        }
        allocations.append(allocation)

        if engine.binding_is_input(i):
            inputs.append(binding)
        else:
            outputs.append(binding)
            
    #print(">>>>>>>>>>>>>>>>>>>>>> 6.1: inputs = ", inputs)
    #print(">>>>>>>>>>>>>>>>>>>>>> 6.2: outputs = ", outputs)

    cuda.memcpy_htod(inputs[0]['allocation'], images)
    context.execute_v2(allocations)   
    inference_outputs = []

    for out in outputs:
        output = np.zeros(out['shape'],out['dtype'])
        cuda.memcpy_dtoh(output, out['allocation'])
        inference_outputs.append(expected_result - output)

        #print(">>>>>>>>>>>>>>>>>>>>>> 7.2: output = ", output)
        #print(">>>>>>>>>>>>>>>>>>>>>> 7.3: expected = ", expected_result)       

    return inference_outputs
        