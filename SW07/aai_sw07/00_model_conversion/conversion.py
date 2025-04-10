from pathlib import Path
import subprocess
import tensorrt as trt

from quanization_calibrators import EntrophyCalibrator, MinMaxCalibrator
from quantization_data_loader import QuantizationDataLoader


def convert_onnx_to_trt_via_trtexec_cli(path_in:Path,path_out:Path,float16:bool):
 
    # trtexec --onnx=<MODELNAME>.onnx --saveEngine=<OUTPUTMODELNAME>.engine --fp16
    command = [
        "/usr/src/tensorrt/bin/trtexec", 
        f"--onnx={path_in.as_posix()}",
        f"--saveEngine={path_out.as_posix()}",
    ]
    
    if float16:
        command.append("--fp16")

    # Run the command
    subprocess.run(command, check=True)
    
    # inspect the model
    # /usr/src/tensorrt/bin/trtexec --loadEngine=<>.trt --dumpLayerInfo > layer_info_fp16.txt

def convert_onnx_to_trt_via_trt_python_api(path_in:Path,path_out:Path,float16:bool):
    
    LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(LOGGER) as builder, \
        builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
        builder.create_builder_config() as config, \
        trt.OnnxParser(network, LOGGER) as parser:
        
        with open(path_in.as_posix(), "rb") as onnx_model:
            parser_success = parser.parse(onnx_model.read())
            if not parser_success:
                for idx in range(parser.num_errors):
                    print(f"ONNX Parsing Error: {parser.get_error(idx)}")
                raise RuntimeError("Failed to parse the ONNX model.")
            
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            if builder.platform_has_fast_fp16:
                for i in range(network.num_layers):
                    layer = network.get_layer(i)
                    #Iterate over layers, assign new types
                    # if layer.type in (trt.LayerType.ANYTHING_WITH_NORMALIZATION):
                    #     layer.precision = trt.float32
            if float16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS) # use your assigned types (if so
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 2 GiB
            
            # requires static imputs -> check netron, click on first node and see if there is any variable, define the sizes in the torch.onnx.export
            
            profile = builder.create_optimization_profile()
            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                
                if input_tensor.shape[0] == -1:  # Check for dynamic batch size (-1)
                    min_shape = [1] + list(input_tensor.shape[1:])  # Min batch size = 1
                    opt_shape = [8] + list(input_tensor.shape[1:])  # Typical batch size
                    max_shape = [16] + list(input_tensor.shape[1:])  # Max batch size
                    
                    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            
            config.add_optimization_profile(profile)
                
            engine = builder.build_engine(network, config)
            if engine is None:
                raise RuntimeError("Failed to build the TensorRT engine.")
            with open(path_out.as_posix(), "wb") as f:
                f.write(engine.serialize())
                
                
def quantize_onnx_to_trt_via_trt_python_api(path_in:Path,path_out:Path,int8:bool,calibrator:str,calibration_stream:QuantizationDataLoader):
    
    LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(LOGGER) as builder, \
        builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
        builder.create_builder_config() as config, \
        trt.OnnxParser(network, LOGGER) as parser:

        with open(path_in.as_posix(), "rb") as onnx_model:
            parser_success = parser.parse(onnx_model.read())
            if not parser_success:
                for idx in range(parser.num_errors):
                    print(f"ONNX Parsing Error: {parser.get_error(idx)}")
                raise RuntimeError("Failed to parse the ONNX model.")
            
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            if builder.platform_has_fast_fp16:
                for i in range(network.num_layers):
                    layer = network.get_layer(i)
                    #Iterate over layers, assign new types
                    # if layer.type in (trt.LayerType.ANYTHING_WITH_NORMALIZATION):
                    #     layer.precision = trt.float32
            if int8:
                config.set_flag(trt.BuilderFlag.INT8)
                calibration_table_path = ""
                if calibrator == "minmax":
                    config.int8_calibrator = MinMaxCalibrator(calibration_stream, calibration_table_path)
                elif calibrator == "entrophy":
                    config.int8_calibrator = EntrophyCalibrator(calibration_stream, calibration_table_path)
                else:
                    config.int8_calibrator = MinMaxCalibrator(calibration_stream, calibration_table_path)
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS) # use your assigned types (if so
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 2 GiB
            
            
            # requires static inputs -> check netron, click on first node and see if there is any variable, define the sizes in the torch.onnx.export
            
            profile = builder.create_optimization_profile()
            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                
                if input_tensor.shape[0] == -1:  # Check for dynamic batch size (-1)
                    min_shape = [1] + list(input_tensor.shape[1:])  # Min batch size = 1
                    opt_shape = [8] + list(input_tensor.shape[1:])  # Typical batch size
                    max_shape = [16] + list(input_tensor.shape[1:])  # Max batch size
                    
                    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            
            config.add_optimization_profile(profile)
                
            
            engine = builder.build_engine(network, config)
            if engine is None:
                raise RuntimeError("Failed to build the TensorRT engine.")
            with open(path_out.as_posix(), "wb") as f:
                f.write(engine.serialize())
                

def convert_onnx_fp32_to_fp16(path_in:Path,path_out:Path):
    import onnx
    from onnxconverter_common import float16 # pip install onnxconverter-common
    
    model = onnx.load(path_in.as_posix())
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, path_out.as_posix())