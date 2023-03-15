import logging
import time
from datetime import datetime

import click
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans, vq, whiten

import clustering.compute as compute
import clustering.utils as utils


@click.command()
@click.option("-i", "--patient_id", type=int, required=True, help="""Patient id""")
@click.option(
    "-m", "--method", type=str, required=False, help="""Clustering method used"""
)
@click.option(
    "-s",
    "--thresold",
    type=float,
    required=True,
    help="""Thresholding values for the binarisation of A""",
)
@click.option(
    "-k",
    "--k_eigen",
    type=int,
    required=True,
    help="""Number of computed eigen values""",
)
def spectral_clustering(patient_id, method, thresold, k_eigen):

    output = utils.get_output_dir()
    logs = output + f"/{datetime.today().strftime('%Y%m%d')}_{patient_id}_logs.txt"
    logging.basicConfig(
        filename=logs,
        level=logging.DEBUG,
        format="%(asctime)s--%(levelname)s-%(message)s",
        filemode="w",
    )
    logging.info(
        f"Clustering - Patient_id: {patient_id} - Date: {datetime.today().strftime('%Y-%m-%d')}"
    )
    logging.info(
        f"Parameters: -i:{patient_id} , -m:{method}, -s:{thresold}, -k:{k_eigen}  "
    )

    logging.info(f"Loading the data... ")
    # load the data
    A = utils.get_A(patient_id)
    logging.info(f"A shape:{A.shape}")

    # Preprocess the data
    A_wm, ind = compute.compute_A_wm(A, patient_id)
    if thresold != 0:
        A_wm, ind = compute.compute_binary_matrix(A_wm, thresold, ind)
    A_wm, ind = compute.compute_fully_connected(A_wm, ind)
    logging.info(f"A_wm shape:{A.shape}")

    # Compute all the required matrix
    L, ind = compute.compute_Lrw(A_wm, ind)
    logging.info(f"L shape:{L.shape}")

    # Compute the eigen vector
    logging.info("Computing the eigen values...")
    v, U, ind = compute.compute_eigenvalues(L, k_eigen, ind)
    v, U = np.real(v), np.real(U)

    logging.info(f"V shape:{U.shape}")

    # Produce the kMean clustering
    V = whiten(U)
    centroids, _ = kmeans(U, k_eigen)
    assignement, dist_ = vq(U, centroids)
    assignement = assignement + 1

    # Export the results
    logging.info("Exporting the results...")
    results = pd.DataFrame(data={"index": ind, "C": assignement})
    results.to_csv(
        output + f"/{datetime.today().strftime('%Y%m%d')}_{patient_id}.txt", index=False
    )
    logging.info("Clustering finished")
    return

    # Convert the results to nifti
    data = utils.load_data(100307)
    f_src_nifti = data["G"]["f"]["mask"]
    f_src_nifti = (
        "/media/miplab-nas2/Data3/Hamid/HCP100_miplabgolgi/" + f_src_nifti[35:]
    )
    cluster_output = "/media/miplab-nas2/Data3/Hamid_Edouard/20230315_100307.txt"
    N_cluster = 10
    output_path = "/media/miplab-nas2/Data3/Hamid_Edouard/test.nii"
    indices_raw = data["G"]["indices"]

    compute.compute_nift(
        f_src_nifti, cluster_output, N_cluster, output_path, indices_raw
    )


if __name__ == "__main__":
    spectral_clustering()
