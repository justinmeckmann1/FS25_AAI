import glob
import os
from pathlib import Path

import cv2
import numpy as np

class QuantizationDataLoader:
    def __init__(self,batch_size:int,calib_image_dir:Path,model_input_height:int,model_input_width:int,model_input_channels:int):
        self.index = 0
        
        self.batch_size = batch_size
        self.model_input_height = model_input_height
        self.model_input_width = model_input_width
        self.model_input_channels = model_input_channels
        self.img_list = glob.glob(calib_image_dir.joinpath("*.png").as_posix())
        print('found all {} images to calib.'.format(len(self.img_list)))
        self.length = len(self.img_list)//self.batch_size
        self.calibration_data = np.zeros((self.batch_size,self.model_input_channels,self.model_input_height,self.model_input_width), dtype=np.float32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), 'not found!!'
                
                # Preprocessing
                
                img = cv2.imread(self.img_list[i + self.index * self.batch_size])
                img = cv2.resize(img, (self.model_input_height, self.model_input_width))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.transpose((2, 0, 1)).astype(np.float32)
                img /= 255.0
                self.calibration_data[i] = img
            self.index += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.length