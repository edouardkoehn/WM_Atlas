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
    type=int,
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
        A_wm = compute.compute_binary_matrix(A_wm, thresold)
    A_wm, ind = compute.compute_fully_connected(A_wm, ind)
    logging.info(f"A_wm shape:{A.shape}")

    # Compute all the required matrix
    D, ind = compute.compute_D(A_wm, ind)
    logging.info(f"D shape:{D.shape}")
    L, ind = compute.compute_L(A_wm, D, ind)
    logging.info(f"L shape:{L.shape}")

    # Compute the eigen vector
    logging.info("Computing the eigen values...")
    W, V, ind = compute.compute_eigenvalues(L, k_eigen, ind)
    V = np.real(V)
    logging.info(f"V shape:{W.shape}")

    # Produce the kMean clustering
    V = whiten(V)
    centroids, _ = kmeans(V, 6)
    assignement, dist_ = vq(V, centroids)

    # Export the results
    logging.info("Exporting the results...")
    results = pd.DataFrame(data={"index": ind, "C": assignement})
    results.to_csv(
        output + f"/{datetime.today().strftime('%Y%m%d')}_{patient_id}.txt", index=False
    )
    logging.info("Clustering finished")

    return


if __name__ == "__main__":
    spectral_clustering()
