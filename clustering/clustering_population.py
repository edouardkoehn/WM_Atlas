import logging
from datetime import datetime

import click
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans, vq, whiten

import clustering.compute as compute
import clustering.utils as utils


@click.command()
@click.option(
    "-i", "--subject_ids", type=int, multiple=True, required=True, help="""Patient id"""
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
    type=click.Choice(["acpc", "nmi"], case_sensitive=False),
    required=True,
    help="""Nifti space used""",
)
@click.option(
    "-s",
    "--save",
    type=bool,
    required=False,
    default=False,
    help="""Saving the intermediate matrix (L, Lrw)""",
)
def clustering_population(
    subject_ids: int,
    method: str = "comb",
    threshold: float = 2,
    k_eigen: int = 10,
    nifti_type: str = "std",
    save: bool = False,
):
    """Workflow to produce the spectral clustering at the population base
    Args:   subject_id(int): coresponding patient id in the database,
            method(str): method used for computing the laplacien,
            threshold(float): thresholding value for the binarisation of the matrix
            k_eigen(int):number of eigen value used
            save(bool):saving the intermediate matrix
    """
    # Define the general paths
    subjects_id = list(subject_ids)
    path_inputs_dir = []
    path_niftis_in = []
    in_dir = utils.get_output_dir()
    work_id = f"/{datetime.today().strftime('%Y%m%d-%H%M')}_{k_eigen}_{threshold}"
    path_logs = f"{utils.create_output_folder(in_dir, subjects_id[0], 'population')}"
    +work_id + "_logs.txt"
    path_nifti_out = f"{utils.create_output_folder(in_dir,subjects_id[0],'population')}"
    +work_id + f"_{nifti_type}.nii.gz"
    path_output_cluster = (
        f"{utils.create_output_folder(in_dir,subjects_id[0],'population')}"
    )
    +work_id + "_clusters.txt"

    for subject_id in subjects_id:
        if utils.check_output_folder:
            path_inputs_dir.append(
                utils.create_output_folder(
                    utils.get_output_dir(), subject_id, "subject"
                )
            )
        else:
            print("Output folder not found : subject_id")

    for output, id in zip(path_inputs_dir, subjects_id):
        path_niftis_in.append(utils.check_nifti(output, method, nifti_type, threshold))
        work_id = f"/{datetime.today().strftime('%Y%m%d-%H%M')}_{id}_{threshold}"

    utils.create_logs(
        path_logs,
        subjects_id,
        datetime.today().strftime("%Y%m%d_%H:%M"),
        method,
        threshold,
        k_eigen,
        nifti_type,
        save,
    )

    # load the data
    Us = []
    dim = (10, 10, 10, 10)
    mask = np.arange(0, 10000).reshape(dim[0], dim[1], dim[2], dim[3])
    mask = np.ravel(mask, order="F")
    mask = mask[0:50]

    for nifti in path_niftis_in:
        U = compute.extract_eigen_from_nifti(nifti, mask)
        Us.append(U)

    # Produce the clustering matrix
    K = np.zeros((len(mask), k_eigen * len(subjects_id)))
    for id in range(0, len(subjects_id)):
        U_subject = Us[id]
        for voxel in range(0, U_subject.shape[0]):
            start_col = id * k_eigen
            end_col = start_col + k_eigen
            K[voxel, start_col:end_col] = U_subject[voxel, 0:k_eigen]

    # Produce the kMean clustering
    K = whiten(K)
    centroids, _ = kmeans(K, k_eigen)
    assignement, dist_ = vq(K, centroids)
    assignement = assignement + 1

    # Export the results
    results = pd.DataFrame(data={"index": mask, "C": assignement})
    if save:
        logging.info("Exporting the results...")
        results.to_csv(path_output_cluster, index=False)
        logging.info("Clustering finished")

    # Convert the results to nifti
    compute.compute_nift(path_niftis_in[0], path_output_cluster, path_nifti_out, save)


if __name__ == "__main__":
    clustering_population()
