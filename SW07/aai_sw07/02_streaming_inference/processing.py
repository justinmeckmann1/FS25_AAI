import numpy as np
import cv2
import time
import torch
import tensorrt as trt
import os
import pycuda.driver as cuda
import pycuda.autoinit

from pathlib import Path

#list of colors for different object ids
colors=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
# font
font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale
fontScale = .6
# line thickness
thickness = 1
#define a limit score to show object
limit_score = 0.1

categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class cifar_model_trt:

    def __init__(self, image_size=32):
        provider = 0

        #engine_path = Path("engines/best_model_dna_0_BS1" + "_pyapi_fp_32.engine")
        engine_path = Path("engines")
        engine_names = os.listdir(engine_path.as_posix())
        engine_names = [file for file in engine_names if file.endswith(".engine")]
        full_engine_path = engine_path.joinpath(engine_names[0])
    
        print('####~~~~'*2, "Streaming with Inference Engine", full_engine_path)
        
        logger = trt.Logger(trt.Logger.ERROR)
        with open(full_engine_path.as_posix(), "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
            self.context =  self.engine.create_execution_context()
            self.inputs = []
            self.outputs = []
            self.allocations = []
            for i in range( self.engine.num_bindings):
                is_input = False
                if  self.engine.binding_is_input(i):
                    is_input = True
                name =  self.engine.get_binding_name(i)
                dtype =  self.engine.get_binding_dtype(i)
                shape =  self.engine.get_binding_shape(i)
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
                self.allocations.append(allocation)
                if  self.engine.binding_is_input(i):
                    self.inputs.append(binding)
                else:
                    self.outputs.append(binding)


        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.image_size = image_size

    def predict(self, frame):
        frame = frame[0:488, 80:560]

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size), 0, 0)
        image = (image.astype(np.float32) / 255. - self.mean) / self.std

        image_t = np.zeros((1, 3, self.image_size, self.image_size), dtype=np.float32)

        for plane in range(3):
            image_t[0, plane] = image[:, :, 2 - plane]

        start = time.time()

        cuda.memcpy_htod(self.inputs[0]['allocation'], image_t)
        self.context.execute_v2(self.allocations)   
        inference_outputs = []
        out = self.outputs[0]
        output = np.zeros(out['shape'],out['dtype'])
        cuda.memcpy_dtoh(output, out['allocation'])

        stop = time.time()

        # convert output to tensor
        prediction = torch.tensor(output)
        # print("predictions ", prediction)

        # we calculate normalised probabilities and average for each category
        normalized_masks = torch.nn.functional.softmax(prediction, dim=1)

        # get precdiction
        class_ind = normalized_masks.argmax(dim=1)
        score = normalized_masks[0, class_ind].item()
        
        img_2_plot = frame.copy()
        category = categories[class_ind]
        cv2.putText(img_2_plot, f'{category}, score={100*score:.1f}', (20,20),
                                    font,  fontScale, colors[class_ind%6], thickness)

        cv2.putText(img_2_plot, f'processing time {1000*(stop - start):.2f} ms', (40, 40), font,
                    fontScale, (255, 255, 255), thickness)

        return img_2_plot




