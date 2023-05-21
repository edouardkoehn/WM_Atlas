import logging

import nibabel as nib
import numpy as np

# Define general path
GENERAL_PATH = "/media/miplab-nas2/Data3/Hamid_Edouard/population_cluster/Bootstrap"
OUTPUT_PATH = GENERAL_PATH + "/heat_map.nii.gz"
PATH_LOGS = GENERAL_PATH + "/heat_map_logs.txt"
logging.basicConfig(
    filename=PATH_LOGS,
    level=logging.INFO,
    format="%(asctime)s--%(levelname)s-%(message)s",
    filemode="w",
)
# Load the data the annotation file
ref_clustering_path = ""
tests_clustering_path = []

vol_ref = nib.load(ref_clustering_path).get_fdata()
vol_ref[np.isnan(vol_ref)] = 0
vol_res = np.zeros_like(vol_ref).astype(int)
n_cluster = 2

for test in tests_clustering_path:
    vol_test = nib.load(test).get_fdata()
    vol_test[np.isnan(vol_test)] = 0
    for cluster in range(1, n_cluster + 1):
        ind_subspace_ref = np.where(vol_ref == cluster)
        subspace_test = np.zeros_like(vol_test).astype(int)
        subspace_test[ind_subspace_ref] = vol_test[ind_subspace_ref]
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
