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


class MELC_Segmentation:
    def __init__(self, data_path, membrane_marker="cd45") -> None:
        self.fields_of_view = sorted(os.listdir(data_path))
        
        self._field_of_view = None
        self._membrane_marker = membrane_marker
        self._model = None
        self._kdforest = dict()        
        self._data_path = data_path
        self._NN_dict = dict()
        
        self.nucleus_label_where = dict()
        self.membrane_label_where = dict()
        
        
    
    @property
    def field_of_view(self):
        return self._field_of_view
    
    
    @field_of_view.setter
    def field_of_view(self, field_of_view):
        if isinstance(field_of_view, str) and field_of_view in self.fields_of_view:
            self._field_of_view = field_of_view    
        else:
            print(field_of_view, "is not valid")
        
    
    def get_membrane_marker(self):
        fov_dir = self.get_fov_dir()
        membrane_marker_path = self._get_channel_path(fov_dir, f"{self._membrane_marker}-")
        return cv2.imread(membrane_marker_path, cv2.IMREAD_GRAYSCALE)

    
    def get_prop_iodide(self):
        fov_dir = self.get_fov_dir()
        bleach_dir = self._get_bleach_dir(fov_dir)
        prop_iodide_path = self._get_channel_path(bleach_dir, "propidium")
        return cv2.imread(prop_iodide_path, cv2.IMREAD_GRAYSCALE)
    
    
    def get_fov_dir(self):
        fov_dir = os.path.join(self._data_path, self._field_of_view)
        assert os.path.isdir(fov_dir), f"Field of view {self._field_of_view} does not exist!"
        return fov_dir
    

    def _get_bleach_dir(self, fov_dir):
        bleach_dir = os.path.join(fov_dir, "bleach")
        assert os.path.isdir(bleach_dir), f"Field of view {fov_dir} does not contain a bleach directory!"
        return bleach_dir
    
    
    def _get_channel_path(self, img_dir, channel):
        channels = [c for c in os.listdir(img_dir) if channel.lower() in c.lower()]
        assert len(channels) == 1, f"Path for field of view {img_dir} does not contain exactly one image of the desired channel!"
        channel_path = os.path.join(img_dir, channels[0])
        return channel_path
   
    def segment(self, img):
        """
        Perform segmentation of a given MELC image
        
        Args:
        - img: MELC image.

        Returns:
        - labels: A numpy array of shape (H, W) representing the predicted label 
                of each pixel in the input image, where H is the image height and 
                W is the image width.
        - cluster_centers: A numpy array of shape (K, 2) representing the (x,y) 
                        coordinates of the cluster centers detected in the input 
                        image, where K is the number of clusters.
        """
        if self._model is None:
            self._model = StarDist2D.from_pretrained('2D_versatile_fluo')     
        axis_norm = (0, 1)
        norm_img = normalize(img, 1,99.8, axis=axis_norm)
        labels, details = self._model.predict_instances(norm_img)
        cluster_centers = details['points']
        return labels, cluster_centers
    
    def radius_from_max_dist_within_segment(self, segment):
        x_min = np.min(segment[0])
        x_max = np.max(segment[0])
        x_diff = x_max - x_min
        y_min = np.min(segment[1])
        y_max = np.max(segment[1])
        y_diff = y_max - y_min
        diff = x_diff if x_diff > y_diff else y_diff
        radius = diff / 2
        return radius
    
    def new_where_nucleus(self, nucleus, nucleus_label):
        if self.field_of_view in self.nucleus_label_where:
            self.nucleus_label_where[self.field_of_view].update({nucleus_label: np.array(nucleus)})
        else:
            self.nucleus_label_where[self.field_of_view] = {nucleus_label: np.array(nucleus)}
            
    def new_where_membrane(self, membrane, membrane_label):
        if self.field_of_view in self.membrane_label_where:
            self.membrane_label_where[self.field_of_view].update({membrane_label: np.array(membrane)})
        else:
            self.membrane_label_where[self.field_of_view] = {membrane_label: np.array(membrane)}


    def existing_membranes_as_nuclei_NN(self, membrane_labels, nuclei_labels, nuclei_centers):
        if self.field_of_view not in self._kdforest:
            self._kdforest[self.field_of_view] = KDTree(nuclei_centers, leafsize=100)
        kdtree = self._kdforest[self.field_of_view]

        reconstructed_membranes = np.zeros_like(membrane_labels, dtype=int)
        nuclei_centers_without_membrane = dict()

        nucleus_radii = list()
        membrane_radii = list()
        nucleus_radii_to_circle = dict()

        for idx, point in enumerate(tqdm(nuclei_centers)):
            membrane_label = membrane_labels[int(point[0]), int(point[1])]
            nucleus_label = nuclei_labels[int(point[0]), int(point[1])]

            if membrane_label > 0: # there is an actual label#
                membrane = np.where(membrane_labels == membrane_label)
                self.new_where_membrane(membrane, membrane_label)
                segment = np.array(membrane).T
            else: 
                nuclei_centers_without_membrane[idx] = point
                nucleus = np.where(nuclei_labels == nucleus_label)
                self.new_where_nucleus(nucleus, nucleus_label)
                nucleus_radii_to_circle[idx] = self.radius_from_max_dist_within_segment(nucleus)
                continue
            dist, idxs = kdtree.query(segment)
            segment = segment[np.where(idxs == idx)].T
                
            nucleus = np.where(nuclei_labels == nucleus_label)
            self.new_where_nucleus(nucleus, nucleus_label)
            # check if membrane is bigger than nucleus:
            if segment.shape[1] < len(nucleus[0]):
                nuclei_centers_without_membrane[idx] = point
                nucleus_radii_to_circle[idx] = self.radius_from_max_dist_within_segment(nucleus)
                continue
            membrane_radii.append(self.radius_from_max_dist_within_segment(segment))
            nucleus_radii.append(self.radius_from_max_dist_within_segment(nucleus))

            reconstructed_membranes[segment[0], segment[1]] = nucleus_label
            self.new_where_membrane(segment, nucleus_label)
            
        ratio = np.array(membrane_radii) / np.array(nucleus_radii)
        radii_ratio = np.median(ratio)
        assert radii_ratio > 1, "Something went wrong with calculating the radii"
        return reconstructed_membranes, nuclei_centers_without_membrane, radii_ratio, nucleus_radii_to_circle

    

    def nearest_neighbors(self, nuclei_centers, nuclei_labels_shape):
        if self.field_of_view not in self._NN_dict:
            kdtree = KDTree(nuclei_centers, leafsize=100)
            xaxis = np.linspace(0, nuclei_labels_shape[0] - 1, nuclei_labels_shape[0])
            yaxis = np.linspace(0, nuclei_labels_shape[1] - 1, nuclei_labels_shape[1])
            xv, yv = np.meshgrid(xaxis, yaxis)
            _, NN_idxs = kdtree.query(np.c_[xv.ravel(), yv.ravel()])
            NN_idxs = NN_idxs.reshape((nuclei_labels_shape[0], nuclei_labels_shape[1])).T 
            self._NN_dict[self.field_of_view] = NN_idxs
        return self._NN_dict[self.field_of_view]
    
    def in_circle(self, x, y, x_c, y_c, radius):
        return ((x - x_c) ** 2 + (y - y_c) ** 2) <= radius ** 2 
    
    def estimate_membranes_as_nuclei_NN_in_radius(self, nuclei_labels, nuclei_centers_without_membrane, radius, nucleus_radii_to_circle):
        if self.field_of_view not in self._kdforest:
            nuclei_centers = list(nuclei_centers_without_membrane.values())
            self._kdforest[self.field_of_view] = KDTree(nuclei_centers, leafsize=100)
        kdtree = self._kdforest[self.field_of_view]
        
        reconstructed_membranes = np.zeros_like(nuclei_labels, dtype=int)
        xaxis = np.linspace(0, nuclei_labels.shape[0] - 1, nuclei_labels.shape[0])
        yaxis = np.linspace(0, nuclei_labels.shape[1] - 1, nuclei_labels.shape[1])

        for idx in tqdm(nuclei_centers_without_membrane):
            point = nuclei_centers_without_membrane[idx]
            nucleus_label = nuclei_labels[int(point[0]), int(point[1])]       

            if idx in nucleus_radii_to_circle:
                nucleus_radius = nucleus_radii_to_circle[idx]
            else:
                nucleus = np.where(nuclei_labels == nucleus_label)
                self.new_where_nucleus(nucleus, nucleus_label)
                nucleus_radius = self.radius_from_max_dist_within_segment(nucleus)

            absolut_radius = radius * nucleus_radius 
            segment = np.array(np.where(self.in_circle(xaxis[:,None], yaxis[None,:], int(point[0]), int(point[1]), np.ceil(absolut_radius) + 1))).T
            dist, idxs = kdtree.query(segment)
            segment = segment[np.where(idxs == idx)].T
            reconstructed_membranes[segment[0], segment[1]] = nucleus_label
            self.new_where_membrane(segment, nucleus_label)
        return reconstructed_membranes 


    
    
    def run(self, field_of_view=None, radii_ratio=None):
        if field_of_view is not None:
            self.field_of_view = field_of_view
        else:
            assert self.field_of_view is not None, "Please specify field of view!"
            
        prop_iodide = self.get_prop_iodide()
        nuclei_labels, nuclei_centers = self.segment(prop_iodide)
        
        if self._membrane_marker is not None:
            membrane_marker = self.get_membrane_marker()
            membrane_labels, _ = self.segment(membrane_marker)
        
            start = time.time()
            reconstructed_membranes, nuclei_centers_without_membranes, radii_ratio, nucleus_radii_to_circle = self.existing_membranes_as_nuclei_NN(membrane_labels, nuclei_labels, nuclei_centers)
            #print("reconstruction took", time.time()-start)
      
        
        else: 
            if radii_ratio is None:
                print("For tissue images, the desired radius needs to be specified as a hyperparameter.")
                return
            nuclei_centers_without_membranes = {i: point for i, point in enumerate(nuclei_centers)}
                
        start = time.time()
        estimated_membranes = self.estimate_membranes_as_nuclei_NN_in_radius(nuclei_labels, nuclei_centers_without_membranes, radii_ratio, nucleus_radii_to_circle)
        #print("estimation took", time.time()-start)
        
        if self._membrane_marker is not None:
            combined_membranes = reconstructed_membranes + estimated_membranes
        else:
            combined_membranes = estimated_membranes
        return nuclei_labels, combined_membranes, self.nucleus_label_where, self.membrane_label_where

    
"""


def existing_membranes_as_nuclei_NN2(self, membrane_labels, nuclei_labels, nuclei_centers):
    NN_idxs = self.nearest_neighbors(nuclei_centers, nuclei_labels.shape)
    reconstructed_membranes = np.zeros_like(membrane_labels, dtype=int)
    nuclei_centers_without_membrane = dict()

    for idx, point in enumerate(nuclei_centers):
        membrane_label = membrane_labels[int(point[0]), int(point[1])]
        if membrane_label == 0:
            nuclei_centers_without_membrane[idx] = point
            continue
        nucleus_label = nuclei_labels[int(point[0]), int(point[1])]
        segment = membrane_labels == membrane_label
        NNs = NN_idxs == idx
        union = NNs * segment
        reconstructed_membranes[np.where(union)] = nucleus_label

    return reconstructed_membranes, nuclei_centers_without_membrane 
    

def estimate_membranes_as_nuclei_NN_in_radius2(self, nuclei_labels, nuclei_centers_without_membrane, radius):
    NN_idxs = self.nearest_neighbors(None, nuclei_labels.shape)
    reconstructed_membranes = np.zeros_like(nuclei_labels, dtype=int)
    xaxis = np.linspace(0, nuclei_labels.shape[0] - 1, nuclei_labels.shape[0])
    yaxis = np.linspace(0, nuclei_labels.shape[1] - 1, nuclei_labels.shape[1])

    n_values, n_counts = np.unique(nuclei_labels, return_counts=True)

    for idx in nuclei_centers_without_membrane:
        point = nuclei_centers_without_membrane[idx]
        nucleus_label = nuclei_labels[int(point[0]), int(point[1])]

        nucleus_radius = np.sqrt(n_counts[np.where(n_values == nucleus_label)][0]) / np.pi
        absolut_radius = radius * nucleus_radius 
        segment = self.in_circle(xaxis[:,None], yaxis[None,:], int(point[0]), int(point[1]), absolut_radius)

        NNs = NN_idxs == idx
        union = NNs * segment
        reconstructed_membranes[np.where(union)] = nucleus_label
    return reconstructed_membranes 


@staticmethod
def estimate_radius(nuclei_labels, membranes_with_nucleus_labels):
    n_values, n_counts = np.unique(nuclei_labels, return_counts=True)
    values, counts = np.unique(membranes_with_nucleus_labels, return_counts=True)
    n_counts = n_counts[np.where(np.isin(n_values, values))]
    assert len(counts) == len(n_counts)

    nuclei_radius = np.sqrt(n_counts[1:] / np.pi)
    membrane_radius = np.sqrt(counts[1:] / np.pi)
    ratio = membrane_radius / nuclei_radius

    median_radius = np.median(ratio)
    return median_radius
"""