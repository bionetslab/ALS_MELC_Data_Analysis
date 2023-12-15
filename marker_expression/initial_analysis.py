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



class ExpressionAnalyzer:
    def __init__(self, data_path, segmentation_results_dir_path, radii_ratio=2, membrane_markers=None, save_plots=False, markers_of_interest=None):
        """
        Initialize the ExpressionAnalyzer class.

        Args:
            data_path (str): The path to the data directory.
            segmentation_results_dir_path (str): The path to the segmentation results directory.
            membrane_markers (list, optional): A list of membrane marker names.
            save_plots (bool, optional): Whether to save generated plots.

        """
        self.data_path = data_path
        self.seg = MELC_Segmentation(data_path, membrane_markers=membrane_markers)
        self.radii_ratio = radii_ratio
        self.expression_data = None
        self.segmentation_results_dir = segmentation_results_dir_path
        self.save_plots = save_plots
        self.markers = dict()
        self.markers_of_interest = markers_of_interest

                 

    def run(self, segment="nuclei", profile=None):
        """
        Run the analysis for expression data.

        Args:
            segment (str, optional): The segment type to analyze (e.g., "nuclei").
            profile (dict, optional): A dictionary defining the expression profile.
            
            {'CD11b-PE': 0, 'CD16-PE': 1, 'CD45RA-PE': 1, 'HLA-DR-PE': 0}

        """
        self.segment_all()
        self.get_expression_of_all_samples(segment)
        if profile is not None:
            self.binarize_and_normalize_expression()
            plot_df = self.count_condition_cells(profile)
            self.plot_condition_df(plot_df, self.title_from_dict(profile), segment)
        
        
    def segment_all(self):
        """
        Segment nuclei and cells for all fields of view.

        """
        for fov in tqdm(self.seg.fields_of_view, desc="Segmenting"):
            nuclei_path = os.path.join(self.segmentation_results_dir, f"{fov}_nuclei.npy")
            if not os.path.exists(nuclei_path):
                try:
                    self.seg.field_of_view = fov
                    nuc, mem, _, _ = self.seg.run(fov, self.radii_ratio)
                    np.save(nuclei_path, nuc.astype(int))
                    np.save(os.path.join(self.segmentation_results_dir, f"{fov}_cells.npy"), mem.astype(int))
                except Exception as e:
                    print(e)
                    continue
                
                
            nuclei_pickle_path = os.path.join(self.segmentation_results_dir, f"{fov}_nuclei.pickle")
            if not os.path.exists(nuclei_pickle_path):
                with open(nuclei_pickle_path, 'wb') as handle:
                    pass
                    #pickle.dump(self.seg.nucleus_label_where[fov], handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(self.segmentation_results_dir, f"{fov}_cell.pickle"), 'wb') as handle:
                    pass
                    #pickle.dump(self.seg.membrane_label_where[fov], handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    def get_expression_per_marker_and_sample(self, adaptive, where_dict):
        """
        Calculate expression for markers and samples.

        Args:
            adaptive (numpy.ndarray): The adaptive thresholded image.
            where_dict (dict): Dictionary mapping labels to coordinates.

        Returns:
            dict: Dictionary of expression values.

        """
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
        """
        Retrieve expression data for all samples.

        Args:
            segment (str): The segment type to analyze (e.g., "nuclei").

        """
        result_dfs = list()

        for fov in tqdm(self.seg.fields_of_view, desc="Calculating expression"):
            os.makedirs(os.path.join(self.segmentation_results_dir, f"marker_expression_{segment}_results/"), exist_ok=True)
            expression_result_path = os.path.join(self.segmentation_results_dir, f"marker_expression_{segment}_results/{fov}.pkl")
            segmentation_result_path = os.path.join(self.segmentation_results_dir, f"{fov}_{segment}.pickle")
            
            self.seg.field_of_view = fov

            markers = {
                m.split("_")[1]: os.path.join(self.seg.get_fov_dir(), m)
                for m in sorted(os.listdir(self.seg.get_fov_dir()))
                if m.endswith(".tif") and "phase" not in m
            }
            keys = list(markers.keys()).copy()
            if self.markers_of_interest:
                for m in keys:
                    col = "-".join(m.split("-")[:-1])
                    if col not in self.markers_of_interest:
                        del markers[m]
    
                    
            self.markers[fov] = markers
            # del markers['Propidium iodide']          

            cols = ["-".join(m.split("-")[:-1]) for m in list(markers.keys())]

            if not os.path.exists(expression_result_path):
                try:
                    with open(segmentation_result_path, 'rb') as handle:
                        where_dict = pickle.load(handle)
                except:
                    print(fov, "did not have segmentation file")
                    continue

                rows = list(where_dict.keys())

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
            
            df.columns = cols
            
            v, c = np.unique(cols, return_counts=True)
            if len(v) != len(markers):
                for m in v[np.where(c > 1)]:
                    max = df[m].max(axis=1).copy()
                    del df[m]
                    df[m] = max


            assert len(df.columns) == len(v)
            
            df["Field of View"] = fov
            df["Sample"] = "_".join(fov.split("_")[0:2])
            df["Group"] = fov.split("_")[0]
            df["Index"] = df.index
            df = df.set_index(["Field of View", "Index"])
            result_dfs.append(df)
        self.expression_data = pd.concat(result_dfs)

    def binarize_and_normalize_expression(self):
        """
        Binarize and normalize expression data.

        """
        control_mean = self.expression_data[self.expression_data["Group"] == "Healthy"].iloc[:, :-2].mean(axis=0)
        control_std = self.expression_data[self.expression_data["Group"] == "Healthy"].iloc[:, :-2].std(axis=0)
        normalized = (self.expression_data.iloc[:, :-2] - control_mean) / control_std
        self.expression_data.iloc[:, :-2] = normalized > 0

    def count_condition_cells(self, profile):
        """
        Count cells based on the specified profile.

        Args:
            profile (dict): A dictionary defining the expression profile.

        Returns:
            pd.DataFrame: DataFrame containing cell counts.

        
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
        """
        pass

    def plot_condition_df(self, plot_df, title, segment):
        """
        Plot the condition DataFrame.

        Args:
            plot_df (pd.DataFrame): The DataFrame to plot.
            title (str): The title for the plot.
            segment (str): The segment type being analyzed.

        """
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
        if self.save_plots:
            filename = title+ "_" + segment + ".pdf"
            print("Saving at", filename)
            plt.savefig(filename)

    @staticmethod
    def title_from_dict(profile):
        """
        Generate a title from a profile dictionary.

        Args:
            profile (dict): A dictionary defining the expression profile.

        Returns:
            str: The generated title.

        """
        title = ""
        for p in profile:
            title += f"{p}:{profile[p]} "
        return title


