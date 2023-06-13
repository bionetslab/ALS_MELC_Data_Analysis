import numpy as np 
import cv2
from tqdm import tqdm 

from stardist.models import StarDist2D

#from csbdeep.utils import normalize
import matplotlib.pyplot as plt
import json
import os
import cv2
import time
import numpy as np
from scipy.spatial import KDTree
from csbdeep.utils import Path, normalize
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pickle


class Expression:
    def __init__(self, seg_res_path, data_path=None, seg=None) -> None:
        if seg is not None:
            assert seg_res_path is not None
            assert data_path is not None
            self.seg_res_path = seg_res_path
            self.seg = seg
        else:
            self.fields_of_view = sorted(os.listdir(data_path))
            markers = {m.split("_")[1]: os.path.join(self.get_fov_dir(fov), m) for m in sorted(os.listdir(self.get_fov_dir())) if m.endswith(".tif") and "phase" not in m}

            self._field_of_view = None    
            self._data_path = data_path
            markers = {m.split("_")[1]: os.path.join(seg.get_fov_dir(), m) for m in sorted(os.listdir(seg.get_fov_dir())) if m.endswith(".tif") and "phase" not in m}

            
            self.nucleus_label_where = dict()
            self.membrane_label_where = dict()
            
        self.expression_dict = dict()
            

            
        
    def _get_channel_path(self, img_dir, channel):
        channels = [c for c in os.listdir(img_dir) if channel.lower() in c.lower()]
        assert len(channels) == 1, f"Path for field of view {img_dir} does not contain exactly one image of the desired channel!"
        channel_path = os.path.join(img_dir, channels[0])
        return channel_path
   
  
        return nuclei_labels, combined_membranes, self.nucleus_label_where, self.membrane_label_where
    

    def get_expression(self, adaptive, where_dict):
        expression = np.zeros_like(adaptive)
        for n in where_dict:
            if n == 0:
                continue
            
            segment = where_dict[n]
            exp = np.sum(adaptive[segment[0], segment[1]])/len(segment[0])
            expression[segment[0], segment[1]] = exp
        return expression

    def get_mean_expression_of_markers(self, markers, reference_group, where_dict, window_size=201):
        # TODO 1 compose markers path with different FOVS 
        if not isinstance(reference_group, list):
            reference_group = [reference_group]
        
        mean_expression = dict()
        for m in markers:
            expressions = list()
            for fov in reference_group:
                #assert fov in self.fields_of_view, f"{fov} cannot be found in the dataset!"
                if fov in self.expression_dict:
                    if m in self.expression_dict[m]:
                        
                else:
                    m_img = cv2.imread(markers[m], cv2.IMREAD_GRAYSCALE) #TODO 1

                    std = np.std(m_img)
                    adaptive = cv2.adaptiveThreshold(m_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, window_size, -std)  
                    
                    expression = self.get_expression(adaptive, where_dict)  
                    self.expression_dict[fov][m] = expression
                    
                expressions.append(expression)
                
            mean_expression[m] = np.mean(expressions)
        
        return mean_expression
        