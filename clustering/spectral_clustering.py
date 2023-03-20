import logging
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
    "-m",
    "--method",
    type=click.Choice(["std", "nrm"], case_sensitive=False),
    required=False,
    default="std",
    help="""Clustering method used (std:Standard(default), nrm:Normalized version)""",
)
@click.option(
    "-t",
    "--threshold",
    type=float,
    default=2,
    required=False,
    help="""Thresholding values for the binarisation of A. If not specified,
    no thresholding would be applied""",
)
@click.option(
    "-k",
    "--k_eigen",
    type=int,
    required=True,
    help="""Number of computed eigen values""",
)
@click.option(
    "-s",
    "--save",
    type=bool,
    required=False,
    default=False,
    help="""Saving the intermediate matrix (L, Lrw)""",
)
def spectral_clustering(
    patient_id: int,
    method: str = "std",
    threshold: float = 2,
    k_eigen: int = 10,
    save: bool = False,
):
    """Workflow to produce the spectral clustering
    Args:   patient_id(int): coresponding patient id in the database,
            method(str): method used for the clustering,
            threshold(float): thresholding value for the binarisation of the matrix
            k_eigen(int):number of eigen value used
            save(bool):saving the intermediate matrix
    """
    # Define the general paths
    path_output_dir = utils.check_output_folder(utils.get_output_dir(), patient_id)
    work_id = f"/{datetime.today().strftime('%Y%m%d-%H%M')}_{patient_id}_{threshold}"
    path_logs = path_output_dir + work_id + "_logs.txt"
    path_cluster = path_output_dir + work_id + "_clusters.txt"
    path_nifti_in = utils.load_data(patient_id)["G"]["f"]["mask"]
    path_nifti_out = path_output_dir + work_id + "_nifti.nii"
    path_matrix = path_output_dir + work_id

    utils.create_logs(
        path_logs,
        patient_id,
        datetime.today().strftime("%Y%m%d_%H:%M"),
        method,
        threshold,
        k_eigen,
        save,
    )
    # load the data
    A = utils.get_A(patient_id)

    # Preprocess the data
    A_wm, ind = compute.compute_A_wm(A, patient_id)
    if threshold != 2:
        A_wm, ind = compute.compute_binary_matrix(A_wm, threshold, ind)
    A_wm, ind = compute.compute_fully_connected(A_wm, ind)

    # Compute all the required matrix
    if method == "std":
        L, ind = compute.compute_L(A_wm, ind, path_matrix, save)
    if method == "nrm":
        L, ind = compute.compute_Lrw(A_wm, ind, path_matrix, save)

    # Compute the eigen vector
    v, U, ind = compute.compute_eigenvalues(L, k_eigen, ind, path_matrix, save)
    v, U = np.real(v), np.real(U)

    # Produce the kMean clustering
    U = whiten(U)
    centroids, _ = kmeans(U, k_eigen)
    assignement, dist_ = vq(U, centroids)
    assignement = assignement + 1

    # Export the results
    logging.info("Exporting the results...")
    results = pd.DataFrame(data={"index": ind, "C": assignement})
    results.to_csv(path_cluster, index=False)
    logging.info("Clustering finished")

    # Convert the results to nifti
    compute.compute_nift(path_nifti_in, path_cluster, path_nifti_out)


if __name__ == "__main__":
    spectral_clustering()
