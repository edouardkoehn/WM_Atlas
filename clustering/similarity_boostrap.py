import logging

import numpy as np
import pandas as pd

from clustering import compute

# Script for computing the similarity between the clustering from a boostrap
# Define general path
GENERAL_PATH = "/media/miplab-nas2/Data3/Hamid_Edouard/population_cluster/Bootstrap"
PATH_ANNOTATION = GENERAL_PATH + "/annotation.csv"
PATH_SIM_MNI = GENERAL_PATH + "/similarity_MNI.csv"
PATH_SIM_CENTRIC = GENERAL_PATH + "/similarity_Centric.csv"
PATH_LOGS = GENERAL_PATH + "/similarity_logs.csv"

logging.basicConfig(
    filename=PATH_LOGS,
    level=logging.INFO,
    format="%(asctime)s--%(levelname)s-%(message)s",
    filemode="w",
)
logging.info("Boostrap similarity:")
# Load the data the annotation file
annotation = pd.read_csv(PATH_ANNOTATION)
annotation = annotation.sort_values(["n_cluster", "batch"]).reset_index(drop=True)

# Extract the path information
experiments = [6]  # to modify depending on the experiments
experiments = [
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
]  # to modify depending on the experiments
logging.info(f"Number of experiments: {len(experiments)}")
logging.info(f"Experiment values: {experiments}")
cluster_paths = []
for exp in experiments:
    cluster_paths.append(
        annotation[annotation.n_cluster == exp].path_nifti_cluster.tolist()
    )
    logging.info(f"Number of batch-{exp}: {len(cluster_paths[-1])}")

# # Compute the MNI distance
# distance_means = []
# distance_std = []
# for exp in range(len(experiments)):
#     logging.info(f"Similarity MNI exp: {experiments[exp]}")
#     distance_matrix = compute.compute_simlarity(cluster_paths[exp], "MNI")
#     distance_means.append(np.mean(distance_matrix, axis=0))
#     logging.info(f"Mean: {distance_means[-1]}")
#     distance_std.append(np.std(distance_matrix, axis=0))
#     logging.info(f"Std: {distance_std[-1]}")

# # Export the mni similarity
# df_distance_mni = pd.DataFrame.from_dict(
#     {
#         "experiement": experiments,
#         "mean_similarity": distance_means,
#         "std_similarity": distance_std,
#     }
# )
# df_distance_mni.to_csv(PATH_SIM_MNI, index=False)

# Compute the element centric distance
distance_means = []
distance_std = []
for exp in range(len(experiments)):
    logging.info(f"Similarity Centric exp: {experiments[exp]}")
    distance_matrix = compute.compute_simlarity(cluster_paths[exp], "Centric")
    distance_means.append(np.mean(distance_matrix, axis=0))
    logging.info(f"Mean: {distance_means[-1]}")
    distance_std.append(np.std(distance_matrix, axis=0))
    logging.info(f"Std: {distance_std[-1]}")

# Export the Centric similarity
df_distance_centric = pd.DataFrame.from_dict(
    {
        "experiement": experiments,
        "mean_similarity": distance_means,
        "std_similarity": distance_std,
    }
)
df_distance_centric.to_csv(PATH_SIM_CENTRIC, index=False)
