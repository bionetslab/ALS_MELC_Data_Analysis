import os
import pickle
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import sys

sys.path.append("/data_nfs/je30bery/ALS_MELC_Data_Analysis/segmentation/")
sys.path.append("/data/bionets/je30bery/ALS_MELC_Data_Analysis/segmentation/")
from melc_segmentation import MELC_Segmentation
from tqdm import tqdm

condition = "melanoma" # choose from ["ctcl", "melanoma"]
seg = MELC_Segmentation(data_path=f"/data/bionets/datasets/melc/{condition}/processed/", membrane_markers=None)
df = pd.DataFrame(index=seg.fields_of_view)    
non_unique_markers = list()

# iterate over all fields of view
for fov in tqdm(seg.fields_of_view, desc="Calculating expression"):
#for fov in tqdm(["Melanoma_29_202006031146_1"], desc="Calculating expression"):
    seg.field_of_view = fov

    # get markers for current field of view
    markers = {
        m.split("_")[1]: os.path.join(seg.get_fov_dir(), m)
        for m in sorted(os.listdir(seg.get_fov_dir()))
        if m.endswith(".tif") and "phase" not in m
    }

    # get names -> remove "FITC", "PE", etc.
    marker_names = ["-".join(m.split("-")[:-1]) for m in markers if len("-".join(m.split("-")[:-1])) > 0]
    # get markers that occure more than once 
    v, c = np.unique(marker_names, return_counts=True)
    double_markers = v[np.where(c > 1)]
    non_unique_markers = non_unique_markers + list(double_markers)

    # calculate pre-processing step and add value to df
    for m in markers:
        if m.split("-")[0] in double_markers:
            m_img = cv2.imread(markers[m], cv2.IMREAD_GRAYSCALE)
            tile_std = np.std(m_img)
            adaptive = cv2.adaptiveThreshold(m_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, -tile_std)
            df.loc[fov, m] = np.sum(np.square(m_img - adaptive)) / m_img.shape[0]**2

df.to_csv(f"effect_of_preprocessing_{condition}.csv")
for m in np.unique(non_unique_markers):
    double_columns = [c for c in df.columns if m in c]
    min = double_columns[df[double_columns].sum(axis=0).argmin()]
    max = double_columns[df[double_columns].sum(axis=0).argmax()]
    if len(double_columns) == 3:
        med = [c for c in double_columns if not (c is max or c is min)]
        assert len(med) == 1
        med = med[0]
        print(f"Choose {min} (MSE = {df[double_columns].sum(axis=0).min()}) over {max} (MSE = {df[double_columns].sum(axis=0).max()}) and {med} (MSE = {df[double_columns].sum(axis=0).median()})")
    else:
        print(f"Choose {min} (MSE = {df[double_columns].sum(axis=0).min()}) over {max} (MSE = {df[double_columns].sum(axis=0).max()})")