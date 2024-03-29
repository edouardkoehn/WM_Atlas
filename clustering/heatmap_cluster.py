import logging

import nibabel as nib
import numpy as np

# Script for the generation of the heatmap based on multiple experiment
# Define general path
GENERAL_PATH = "/media/miplab-nas2/Data3/Hamid_Edouard/population_cluster/Bootstrap"
OUTPUT_PATH = GENERAL_PATH + "/heat_map.nii.gz"
PATH_LOGS = GENERAL_PATH + "/heat_map_logs.txt"
src_path = "/media/miplab-nas2/Data3/Hamid_Edouard/population_cluster/Bootstrap/"
logging.basicConfig(
    filename=PATH_LOGS,
    level=logging.INFO,
    format="%(asctime)s--%(levelname)s-%(message)s",
    filemode="w",
)

# Load the data the annotation file, parameter to modify
n_cluster = 10
ref_clustering_path = f"{src_path}20230610-0910_12_2.0_92_10_10_reslice_cluster.nii.gz"
tests_clustering_path = [
    f"{src_path}20230603-0744_12_2.0_25_10_1_reslice_cluster.nii.gz",
    f"{src_path}20230603-0744_12_2.0_25_10_10_reslice_cluster.nii.gz",
    f"{src_path}20230603-0744_12_2.0_25_10_2_reslice_cluster.nii.gz",
    f"{src_path}20230603-0744_12_2.0_25_10_3_reslice_cluster.nii.gz",
    f"{src_path}20230603-0744_12_2.0_25_10_4_reslice_cluster.nii.gz",
    f"{src_path}20230603-0744_12_2.0_25_10_5_reslice_cluster.nii.gz",
    f"{src_path}20230603-0744_12_2.0_25_10_6_reslice_cluster.nii.gz",
    f"{src_path}20230603-0744_12_2.0_25_10_7_reslice_cluster.nii.gz",
    f"{src_path}20230603-1718_12_2.0_25_10_8_reslice_cluster.nii.gz",
    f"{src_path}20230603-0744_12_2.0_25_10_9_reslice_cluster.nii.gz",
]


vol_ref = nib.load(ref_clustering_path).get_fdata()
vol_ref[np.isnan(vol_ref)] = 0
vol_res = np.zeros_like(vol_ref).astype(int)

logging.info(f"Number of clusters: {n_cluster}")
logging.info(f"Ref clustering: {ref_clustering_path}")
logging.info(f"Tests clustering: {tests_clustering_path}")
for test in tests_clustering_path:
    logging.info(f"Proccesing test :{test}")
    vol_test = nib.load(test).get_fdata()
    vol_test[np.isnan(vol_test)] = 0
    for cluster in range(1, n_cluster + 1):
        ind_subspace_ref = np.where(vol_ref == cluster)
        subspace_test = np.zeros_like(vol_test).astype(int)
        subspace_test[ind_subspace_ref] = vol_test[ind_subspace_ref]
        print(subspace_test[ind_subspace_ref].shape)
        biggest_assignement_subspace_test = np.argmax(
            np.bincount(subspace_test[ind_subspace_ref].ravel())
        )
        subspace_test = np.where(
            subspace_test == biggest_assignement_subspace_test, 1, 0
        )
        vol_res = vol_res + subspace_test

vol_res = vol_res / len(tests_clustering_path)
h = nib.load(ref_clustering_path)
hd = h.header
heat_map = nib.Nifti1Image(vol_res, None, header=hd)
nib.save(heat_map, OUTPUT_PATH)
