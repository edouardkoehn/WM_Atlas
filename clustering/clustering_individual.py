import logging
from datetime import datetime

import click
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans, vq, whiten

import clustering.compute as compute
import clustering.utils as utils
import clustering.utils_nifti as utils_nifti


@click.command()
@click.option(
    "-i", "--subject_id", type=int, multiple=False, required=True, help="""Patient id"""
)
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
    "-n",
    "--nifti_type",
    type=click.Choice(["acpc", "mni"], case_sensitive=False),
    required=True,
    help="""Nifti space used""",
)
@click.option(
    "-v",
    "--value_type",
    type=click.Choice(["cluster", "distance", "z_score"], case_sensitive=False),
    required=False,
    help="""Value to assign in the volume""",
    multiple=True,
)
@click.option(
    "-s",
    "--save",
    type=bool,
    required=False,
    default=False,
    help="""Saving the intermediate matrix (L, Lrw)""",
)
def clustering_individual(
    subject_id: int,
    method: str = "comb",
    threshold: float = 2,
    k_eigen: int = 10,
    nifti_type: str = "nmi",
    value_type: str = "cluster",
    save: bool = False,
):
    """Workflow to produce the spectral clustering at the population base
    Args:   subject_id(int): coresponding patient id in the database,
            method(str): method used for computing the laplacien,
            threshold(float): thresholding value for the binarisation of the matrix
            k_eigen(int):number of eigen value used
            nifti_type(str): from which nifti space, you want to produce the clustering
            save(bool):saving the intermediate matrix
    """
    # Define the general paths
    values_type = list(value_type)
    in_dir = utils.get_output_dir()
    work_id = (
        f"/{datetime.today().strftime('%Y%m%d-%H%M')}_{subject_id}_{threshold}"
        + f"_{k_eigen}"
    )
    path_logs = (
        f"{utils.create_output_folder(in_dir, subject_id, 'subject')}"
        + work_id
        + "_clustering_logs.txt"
    )
    path_nifti_out = [
        (
            f"{utils.create_output_folder(in_dir,subject_id,'subject')}"
            + work_id
            + f"_clusters_{nifti_type}_{value}.nii.gz"
        )
        for value in value_type
    ]

    path_output_cluster = (
        f"{utils.create_output_folder(in_dir,subject_id,'subject')}"
        + work_id
        + "_clusters.txt"
    )
    if utils.check_output_folder:
        path_input_dir = utils.create_output_folder(
            utils.get_output_dir(), subject_id, "subject"
        )
    else:
        print("Output folder not found : subject_id")
    path_nifti_in = utils.check_nifti(path_input_dir, method, nifti_type, threshold)

    utils.create_logs(
        path_logs,
        subject_id,
        datetime.today().strftime("%Y%m%d_%H:%M"),
        method,
        threshold,
        k_eigen,
        nifti_type,
        values_type,
        save,
    )

    # Load the data
    wm_indices = []
    U, wm_indices = utils_nifti.extract_eigen_from_nifti(
        path_nifti_in, wm_indices, "individual"
    )

    # Produce the clustering matrix
    K = np.zeros((len(U), k_eigen))
    for voxel in range(0, U.shape[0]):
        K[voxel, 0:k_eigen] = U[voxel, 0:k_eigen]
    logging.info(f"Clustering matrix shape : {K.shape}")

    # Produce the kMean clustering
    K = whiten(K)
    centroids, dist = kmeans(K, k_eigen)
    assignements, dist_ = vq(K, centroids)
    dist = compute.compute_distance(centroids, K, assignements)
    z_score = compute.compute_zscore(centroids, K, assignements)
    assignements = assignements + 1

    # Export the results
    results = pd.DataFrame(
        data={"index": wm_indices, "C": assignements, "dist": dist, "z_score": z_score}
    )
    if save:
        logging.info("Exporting the results...")
        results.to_csv(path_output_cluster, index=False)
        logging.info("Clustering finished")

    # Convert the results to nifti
    for value, path_out in zip(values_type, path_nifti_out):
        utils_nifti.compute_nift(
            path_nifti_in, path_output_cluster, path_out, value, save
        )


if __name__ == "__main__":
    clustering_individual()
