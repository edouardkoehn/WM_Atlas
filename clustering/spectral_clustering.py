import logging
from datetime import datetime

import click
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans, vq, whiten

import clustering.compute as compute
import clustering.utils as utils


@click.command()
@click.option("-i", "--subject_id", type=int, required=True, help="""Patient id""")
@click.option(
    "-m",
    "--method",
    type=click.Choice(["comb", "rw"], case_sensitive=False),
    required=False,
    default="std",
    help="""Clustering method used (comb:Combinatorial(default),
    rd:Random Walke version)""",
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
    subject_id: int,
    method: str = "std",
    threshold: float = 2,
    k_eigen: int = 10,
    save: bool = False,
):
    """Workflow to produce the spectral clustering
    Args:   subject_id(int): coresponding patient id in the database,
            method(str): method used for computing the laplacien,
            threshold(float): thresholding value for the binarisation of the matrix
            k_eigen(int):number of eigen value used
            save(bool):saving the intermediate matrix
    """
    # Define the general paths
    path_output_dir = utils.create_output_folder(utils.get_output_dir(), subject_id)
    work_id = f"/{datetime.today().strftime('%Y%m%d-%H%M')}_{subject_id}_{threshold}"
    path_logs = path_output_dir + work_id + "_logs.txt"
    path_cluster = path_output_dir + work_id + "_clusters.txt"
    path_nifti_in = utils.load_data(subject_id)["G"]["f"]["mask"]
    path_nifti_out = path_output_dir + work_id + "_nifti.nii"
    path_matrix = path_output_dir + work_id

    utils.create_logs(
        path_logs,
        subject_id,
        datetime.today().strftime("%Y%m%d_%H:%M"),
        method,
        threshold,
        k_eigen,
        save,
    )
    # load the data
    A = utils.get_A(subject_id)

    # Preprocess the data
    A_wm, ind = compute.compute_A_wm(A, subject_id)

    # A_wm=A_wm[0:1000,0:1000]
    # ind=ind[0:1000]
    if threshold != 2:
        A_wm, ind = compute.compute_binary_matrix(A_wm, threshold, ind)
    A_wm, ind = compute.compute_fully_connected(A_wm, ind)

    # Compute all the required matrix
    if method == "comb":
        L, ind = compute.compute_L(A_wm, ind, path_matrix, save)
    if method == "rw":
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
    compute.compute_nift(path_nifti_in, path_cluster, path_nifti_out, save)


if __name__ == "__main__":
    spectral_clustering()
