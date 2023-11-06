import matplotlib.pyplot as plt
import json
import os
import cv2
import time
import numpy as np
from csbdeep.utils import Path, normalize
import pandas as pd
import anndata as ad
from tqdm import tqdm
import pickle
import sys
sys.path.append("/data/bionets/je30bery/ALS_MELC_Data_Analysis/segmentation/")
sys.path.append("/data/bionets/je30bery/ALS_MELC_Data_Analysis/marker_expression/")
from segmentation import MELC_Segmentation
from initial_analysis import ExpressionAnalyzer
import anndata as ad
import warnings
warnings.filterwarnings("ignore")


data = "CTCL"

f = open("/data/bionets/je30bery/ALS_MELC_Data_Analysis/config.json")
config = json.load(f)
data_path = config[data]
seg_results_path = config["segmentation_results"]
os.makedirs(os.path.join(seg_results_path, "anndata_files"), exist_ok=True)
seg = MELC_Segmentation(data_path, membrane_markers=None) 
# membrane_marker: str/None 
# radius: multiple of cell radius
comorbidity_info = False

antibody_gene_symbols = {
    'ADAM10-FITC': 'ADAM10',
    'ADAM10-PE': 'ADAM10',
    'CD107a-FITC': 'LAMP1',
    'CD11a-PE': 'ITGAL',
    'CD11c-PE': 'ITGAX',
    'CD123-FITC': 'IL3RA',
    'CD138-FITC': 'SDC1',
    'CD14-PE': 'LBP',
    'CD16-PE': 'FCGR3A',
    'CD163-PE': 'CD163',
    'CD19-PE': 'CD19',
    'CD1a-PE': 'CD1A',
    'CD20-FITC': 'MS4A1',
    'CD205-PE': 'LY75',
    'CD206-FITC': 'MRC1',
    'CD209-FITC': 'CD209',
    'CD24-FITC': 'CD24',
    'CD25-PE': 'IL2RA',
    'CD271-FITC': 'NGFR',
    'CD29-FITC': 'ITGB1',
    'CD3-PE': 'CD3E',
    'CD36-FITC': 'CD36',
    'CD38-PE': 'CD38',
    'CD4-PE': 'CD4',
    'CD40-PE': 'CD40',
    'CD44-PE': 'CD44',
    'CD45-PE': 'PTPRC',
    'CD45RA-PE': 'PTPRC',
    'CD45RO-FITC': 'PTPRC',
    'CD52-FITC': 'CD52',
    'CD54-FITC': 'ICAM1',
    'CD55-PE': 'CD55',
    'CD56-PE': 'NCAM1',
    'CD6-PE': 'CD6',
    'CD62P-PE': 'SELP',
    'CD63-FITC': 'CD63',
    'CD68-FITC': 'CD68',
    'CD69-PE': 'CD69',
    'CD71-FITC': 'TFRC',
    'CD8-PE': 'CD8A',
    'CD9-PE': 'CD9',
    'CD95-PE': 'TNFRSF6',
    'Collagen IV-FITC': 'COL4A1',
    'Cytokeratin-14-FITC': 'KRT14',
    'E-Cadherin-FITC': 'CDH1',
    'EGFR-AF488': 'EGFR',
    'HLA-ABC-PE': 'HLA-A, HLA-B, HLA-C',
    'HLA-DR-PE': 'HLA-DRB1',
    'KIP1-FITC': 'CDKN1B',
    'Melan-A-FITC': 'MLANA',
    'Nestin-AF488': 'NES',
    'Notch-1-FITC': 'NOTCH1',
    'Notch-2-PE': 'NOTCH2',
    'Notch-3-PE': 'NOTCH3',
    'Notch-4-PE': 'NOTCH4',
    'PPARgamma-FITC': 'PPARG',
    'PPB-FITC': 'VIM',
    'TACE-FITC': 'ADAM17',
    'TAP73-FITC': 'TP73',
    'TNFR1-PE': 'TNFRSF1A',
    'TNFR2-PE': 'TNFRSF1B',
    'Vimentin-FITC': 'VIM',
    'beta-Catenin-FITC': 'CTNNB1',
    'p63-FITC': 'TP63',
    'phospho-Connexin-FITC': 'GJA1',
    'Melan-A-AF488': 'MLANA',
    'Bcl-2-FITC': 'BCL2',
    'CD10-FITC': 'CD10',
    'CD11b-FITC': 'ITGAM',
    'CD13-FITC': 'ANPEP',
    'CD141-PE': 'THBD',
    'CD15-FITC': 'FUT4',
    'CD2-FITC': 'CD2',
    'CD27-PE': 'CD27',
    'CD276-FITC': 'B7-H3',
    'CD43-PE': 'SPN',
    'CD5-PE': 'CD5',
    'CD53-FITC': 'F11R',
    'CD7-PE': 'CD7',
    'CD81-FITC': 'CD81',
    'CD83-FITC': 'CD83',
    'CD90-FITC': 'THY1',
    'CD146-FITC': 'MCAM',
    'CD34-FITC': 'CD34',
    'CD39-FITC': 'ENTPD1',
    'CD56-FITC': 'NCAM1',
    'CD58-FITC': 'CD58',
    'CD66abce-FITC': 'CEACAM',
    'CD73-FITC': 'NT5E',
    'CD80-PE': 'CD80',
    'L302-FITC': 'CD302',
    'CD141-FITC': 'THBD',
    'APPC-FITC': 'CD172a',
    'Actin-FITC': 'ACTB',
    'BOP-FITC': 'TSPAN7',
    'C97-FITC': 'CD97',
    'CALCA-FITC': 'CALCA',
    'CEB2-FITC': 'CEBPB',
    'CEB6-FITC': 'CEBPE',
    'CK2-A1-FITC': 'CSNK2A1',
    'CK2-A2-FITC': 'CSNK2A2',
    'CPB3-FITC': 'CPB3',
    'Caspase 3 active-FITC': 'CASP3',
    'Cyclin D1-FITC': 'CCND1',
    'Cytochrome C-FITC': 'CYCS',
    'Cytokeratin-15-DyLight488': 'KRT15',
    'DKK3-FITC': 'DKK3',
    'DRO-FITC': 'DROSHA',
    'Desmin-FITC': 'DES',
    'EBF-P-FITC': 'EBF1',
    'EK2-FITC': 'KRT8',
    'ERG1-FITC': 'ERG',
    'FCepsilonRIa-FITC': 'FCER1A',
    'FLT-FITC': 'FLT3',
    'FRP 13E12-FITC': 'HLA-DRB1',
    'FST-FITC': 'FST',
    'KIS-FITC': 'MKI67',
    'Ki67-FITC': 'MKI67',
    'LOC-HE-FITC': 'NHEJ1',
    'LPAM-1-FITC': 'ITGAE',
    'LSDP-FITC': 'GFER',
    'MCSP-FITC': 'CSPG4',
    'MECP2-FITC': 'MECP2',
    'MPV-FITC': 'MYH9',
    'NKp80-FITC': 'KLRC1',
    'NRG9-FITC': 'NRG1',
    'Neutrophil Elastase-FITC': 'ELANE',
    'ORC2-FITC': 'ORC2',
    'ORC3-FITC': 'ORC3',
    'PBS': 'Phosphate buffered salt solution',
    'PCNA-FITC': 'PCNA',
    'PGRN-FITC': 'GRN',
    'PRPB-FITC': 'PRPF8',
    'RIK-2-FITC': 'ETS1',
    'RIM3-FITC': 'RIMS3',
    'Reelin-FITC': 'RELN',
    'STAT3-FITC': 'STAT3',
    'SYT10-FITC': 'SYT10',
    'TDP-FITC': 'TARDBP',
    'beta-Tubulin-FITC': 'TUBB',
    'p53-FITC': 'TP53',
    'CD163-FITC': 'CD163',
    'Ki67-AF488': 'MKI67',
    'CD20-PE': 'MS4A1',
    'CD3-FITC': 'CD3E',
    'CD303-PE': 'BDCA-2 (CD303)',
    'CD31-FITC': 'PECAM1 (CD31)',
    'CD46-FITC': 'CD46',
    'CD62L-FITC': 'SELL (CD62L)',
    'CD71-PE': 'TFRC',
    'CLA-FITC': 'CLA',
    'FoxP3-FITC': 'FOXP3',
    'IgA-FITC': 'IGHG',
    'TcR alpha': 'TRAC',
    'CD117-PE': 'KIT',
    'CD2-PE': 'CD2',
    'CD31-PE': 'PECAM1 (CD31)',
    'CD34-PE': 'CD34',
    'CD68-PE': 'CD68',
    'CD73-PE': 'NT5E',
    'Follicular Dendritic Cells-FITC': 'CD21',
    'FoxP3-PE': 'FOXP3',
    'III beta-Tubulin-FITC': 'TUBB3',
    'TSLP-PE': 'TSLP'
}



os.makedirs(os.path.join(seg_results_path, "anndata_files"), exist_ok=True)

for segment in ["nuclei", "cell"]:
    result_dict = dict()
    
    if comorbidity_info:
        comorbidities = pd.read_csv("/data_slow/je30bery/data/ALS/ALS_comorbidities.txt", delimiter=";")
        comorbidities = comorbidities.set_index("pat_id")

    EA = ExpressionAnalyzer(data_path=data_path, segmentation_results_dir_path=seg_results_path, membrane_markers=None)
    EA.run(segment=segment, profile=None)
    expression_data = EA.expression_data.sort_index()
    expression_data = expression_data.fillna(0)
    #expression_data = expression_data.drop_duplicates()

    for i, fov in enumerate(tqdm(seg.fields_of_view)):
        #if os.path.exists(os.path.join(seg_results_path, "anndata_files", f"adata_{segment}_{fov}.pickle")):
         #   continue
        
        if "ipynb" in fov:
            continue

        seg.field_of_view = fov
        if os.path.exists(os.path.join(seg_results_path, f"{fov}_nuclei.pickle")):
            
            with open(os.path.join(seg_results_path, f"{fov}_nuclei.pickle"), "rb") as handle:
                where_nuc = pickle.load(handle)
            with open(os.path.join(seg_results_path, f"{fov}_cell.pickle"), "rb") as handle:
                where_cell = pickle.load(handle)
            nuc = np.load(os.path.join(seg_results_path, f"{fov}_nuclei.npy"))
            cell = np.load(os.path.join(seg_results_path, f"{fov}_cells.npy"))
        else:
            nuc, cell, where_nuc, where_cell = seg.run()

        where_dict = where_nuc if segment == "nuclei" else where_cell   
        where_dict = dict(sorted(where_dict.items()))
        
        markers = {
            m.split("_")[1]: os.path.join(seg.get_fov_dir(), m)
            for m in sorted(os.listdir(seg.get_fov_dir()))
            if m.endswith(".tif") and "phase" not in m
        }
        
        
        group =  np.unique(expression_data.loc[fov]["Group"].astype(str).values)[0]
        pat_id = np.unique(expression_data.loc[fov]["Sample"].astype(str).values)[0]

        exp_fov = expression_data.loc[fov].copy()
        exp_fov = exp_fov.drop(["Sample", "Group"], axis=1)
        
        adata = ad.AnnData(exp_fov)
        adata.var = pd.DataFrame(np.array([antibody_gene_symbols[antibody] for antibody in exp_fov.columns]), columns=["gene_symbol"])
        adata.obsm["cellLabelInImage"] = np.array([int(a) for a in list(exp_fov.index)])

        adata.varm["antibody"] = pd.DataFrame(exp_fov.columns, columns=["antibody"])
        adata.obsm["cellSize"] = np.array([len(where_dict[k][0]) for k in where_dict])      
        adata.obsm["Group"] = np.array([group] * len(adata.obsm["cellSize"]))
                
        adata.uns["patient_id"] = pat_id

        adata.obsm["patient_label"] = np.array([pat_id] * len(adata.obsm["cellSize"]))


        if comorbidity_info:
            """
            for c in comorbidities.columns:
                if "ALS" in sample:
                    adata.obsm[str(c)] = np.array([str(comorbidities.loc[sample, c])]* exp_fov.shape[0])
                    adata.uns[str(c)] = str(comorbidities.loc[sample, c])
                else:
                    adata.obsm[str(c)] = np.array(["unknown"] * exp_fov.shape[0])
                    adata.uns[str(c)] = "unknown"
            """
            pass
    
        adata.obsm["field_of_view"] = np.array([fov] * exp_fov.shape[0]) 
        #adata.uns["field_of_view"] = fov
        
    
        if "spatial" not in adata.uns:
            adata.uns["spatial"] = {}  # Create the "spatial" key if it doesn't exist

        if "images" not in adata.uns["spatial"]:
            adata.uns["spatial"]["images"] = {}  # Create the "images" key if it doesn't exist
        
        adata.uns["spatial"]["images"]["Propidium iodide"] = seg.get_prop_iodide()
        for m in markers:
            adata.uns["spatial"]["images"][str(m)] = cv2.imread(markers[m], cv2.IMREAD_GRAYSCALE)
    
        adata.obsm["control_mean_expression"] = np.array([EA.expression_data[EA.expression_data["Group"] == "Healthy"].iloc[:, :-2].mean(axis=0).values] * exp_fov.shape[0])
        adata.obsm["control_std_expression"] = np.array([EA.expression_data[EA.expression_data["Group"] == "Healthy"].iloc[:, :-2].std(axis=0).values] * exp_fov.shape[0])       
        adata.uns["control_mean_expression"] = EA.expression_data[EA.expression_data["Group"] == "Healthy"].iloc[:, :-2].mean(axis=0).values
        adata.uns["control_std_expression"] = EA.expression_data[EA.expression_data["Group"] == "Healthy"].iloc[:, :-2].std(axis=0).values


        adata.uns["cell_coordinates"] = where_dict
        adata.uns["spatial"]["segmentation"] = nuc if segment == "nuclei" else cell

        result_dict[i] = adata
        
    with open(os.path.join(seg_results_path, "anndata_files", f"adata_{segment}.pickle"), 'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

