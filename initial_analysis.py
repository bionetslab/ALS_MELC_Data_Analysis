import os
import pickle
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from segmentation import MELC_Segmentation
from tqdm import tqdm

class ExpressionAnalyzer:
    def __init__(self, data_path="/data_slow/je30bery/data/ALS", segmentation_results_dir="/data_slow/je30bery/spatial_proteomics/segmentation_results/", membrane_marker="cd45"):
        self.data_path = data_path
        self.membrane_marker = membrane_marker
        self.seg = MELC_Segmentation(data_path, membrane_marker=membrane_marker)
        self.expression_data = None
        self.segmentation_results_dir = segmentation_results_dir
                 

    def run(self, segment="nuclei", profile={'CD11b-PE': 0, 'CD16-PE': 1, 'CD45RA-PE': 1, 'HLA-DR-PE': 0}):
        self.segment_all()
        self.get_expression_of_all_samples(segment)
        self.binarize_and_normalize_expression()
        plot_df = self.count_condition_cells(profile)
        self.plot_condition_df(plot_df, self.title_from_dict(profile), segment)

        
    def segment_all(self):
        for i, fov in enumerate(tqdm(self.seg.fields_of_view, desc="Segmenting")):
            nuclei_path = os.path.join(self.segmentation_results_dir, f"{fov}_nuclei.npy")
            if not os.path.exists(nuclei_path):
                self.seg.field_of_view = fov
                nuc, mem, _, _ = self.seg.run(fov)
                np.save(nuclei_path, nuc.astype(int))
                np.save(os.path.join(self.segmentation_results_dir, f"{fov}_cells.npy"), mem.astype(int))

            nuclei_pickle_path = os.path.join(self.segmentation_results_dir, f"{fov}_nuclei.pickle")
            if not os.path.exists(nuclei_pickle_path):
                with open(nuclei_pickle_path, 'wb') as handle:
                    pickle.dump(self.seg.nucleus_label_where[fov], handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(self.segmentation_results_dir, f"{fov}_cell.pickle"), 'wb') as handle:
                    pickle.dump(self.seg.cell_label_where[fov], handle, protocol=pickle.HIGHEST_PROTOCOL)

                    
    def get_expression_per_marker_and_sample(self, adaptive, where_dict):
        expression = np.zeros_like(adaptive)
        expression_dict = dict()
        for n in where_dict:
            if n == 0:
                continue

            segment = where_dict[n]
            exp = np.sum(adaptive[segment[0], segment[1]]) / len(segment[0])
            expression_dict[n] = exp / 255
        return expression_dict

    
    def get_expression_of_all_samples(self, segment):
        result_dfs = list()

        for fov in tqdm(self.seg.fields_of_view, desc="Calculating expression"):
            expression_result_path = f"./marker_expression_{segment}_results/{fov}.pkl"
            segmentation_result_path = os.path.join(self.segmentation_results_dir, f"{fov}_{segment}.pickle")
                                                    
            if not os.path.exists(expression_result_path):
                with open(segmentation_result_path, 'rb') as handle:
                    where_dict = pickle.load(handle)

                self.seg.field_of_view = fov
                markers = {
                    m.split("_")[1]: os.path.join(self.seg.get_fov_dir(), m)
                    for m in sorted(os.listdir(self.seg.get_fov_dir()))
                    if m.endswith(".tif") and "phase" not in m
                }
                del markers['Propidium iodide']
                rows = list(where_dict.keys())
                cols = list(markers.keys())

                df = pd.DataFrame(index=rows, columns=markers)
                for m in markers:
                    m_img = cv2.imread(markers[m], cv2.IMREAD_GRAYSCALE)
                    tile_std = np.std(m_img)
                    adaptive = cv2.adaptiveThreshold(m_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, -tile_std)

                    expression_dict = self.get_expression_per_marker_and_sample(adaptive, where_dict)
                    assert list(where_dict.keys()) == list(expression_dict.keys())
                    df[m] = list(expression_dict.values())

                df.to_pickle(expression_result_path)

            else:
                df = pd.read_pickle(expression_result_path)

            df["Field of View"] = fov.split(".")[0]
            df["Sample"] = fov.split(" ")[0]
            df["Group"] = "Case" if "ALS" in fov else "Control"
            temp = df.index
            df["Index"] = df.index
            df = df.set_index(["Field of View", "Index"])
            result_dfs.append(df)
        self.expression_data = pd.concat(result_dfs)

    def binarize_and_normalize_expression(self):
        control_mean = self.expression_data[self.expression_data["Group"] == "Control"].iloc[:, :-2].mean(axis=0)
        control_std = self.expression_data[self.expression_data["Group"] == "Control"].iloc[:, :-2].std(axis=0)
        normalized = (self.expression_data.iloc[:, :-2] - control_mean) / control_std
        self.expression_data.iloc[:, :-2] = normalized > 0

    def count_condition_cells(self, profile):
        plot_df = pd.DataFrame(columns=['Counts', "Sample", "Group"])
        condition_df = self.expression_data
        for p in profile:
            if profile[p] == 0:
                condition_df = condition_df[condition_df[p] == 0]
            else:
                condition_df = condition_df[condition_df[p] > 0]

        for i, sample in enumerate(np.unique(condition_df["Sample"])):
            condition_cells = len(condition_df[condition_df["Sample"] == sample])
            total_cells = len(self.expression_data[self.expression_data["Sample"] == sample])
            group = "Case" if "ALS" in sample else "Control"
            plot_df.loc[i] = [condition_cells / total_cells * 100, sample, group]
        return plot_df

    def plot_condition_df(self, plot_df, title, segment):
        plt.clf()
        sns.set_theme()
        patients = len(plot_df[plot_df["Group"] == "Case"])
        pal = sns.color_palette("rocket", patients) + sns.color_palette("mako", len(plot_df) - patients)
        sns.scatterplot(plot_df, x="Group", y="Counts", hue="Sample", palette=pal)
        plt.ylabel("Cell Count [%]")
        #plt.legend(bbox_to_anchor=(1, 1))
        plt.title(title)
        plt.suptitle(f"Expression in {segment.upper()[0]}{segment[1:]}")
        plt.tight_layout()
        plt.legend(loc="center")
        
        filename = title+ "_" + segment + ".pdf"
        print("Saving at", filename)
        plt.savefig(filename)

    @staticmethod
    def title_from_dict(profile):
        title = ""
        for p in profile:
            title += f"{p}:{profile[p]} "
        return title


