import logging
import os
from datetime import datetime

import click
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans, vq, whiten

import clustering.compute as compute
import clustering.utils as utils
import clustering.utils_nifti as nifti_utils


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
    "-c",
    "--clusters",
    type=int,
    required=False,
    default=0,
    help="""Number of clusters""",
)
@click.option(
    "-n",
    "--nifti_type",
    type=click.Choice(["reslice"], case_sensitive=False),
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
    "-b",
    "--batch_number",
    type=int,
    required=False,
    help=""""Batch number""",
    default=0,
)
@click.option(
    "-s",
    "--save",
    type=bool,
    required=False,
    default=False,
    help="""Saving the intermediate matrix (L, Lrw)""",
)
def clustering_boostrap(
    subject_ids: int,
    method: str = "comb",
    threshold: float = 2,
    k_eigen: int = 10,
    nifti_type: str = "nmi",
    value_type: str = "cluster",
    batch_number: int = 0,
    save: bool = False,
    clusters: int = 0,
):
    """Workflow to produce the spectral clustering at the population base
    Args:   subject_id(int): coresponding patient id in the database,
            method(str): method used for computing the laplacien,
            threshold(float): thresholding value for the binarisation of the matrix
            k_eigen(int):number of eigen value used
            nifti_type:from which nifti the clustering would be produced
            save(bool):saving the intermediate matrix
    """
    # Define the general paths
    if clusters == 0:
        clusters = k_eigen

    subjects_id = list(subject_ids)
    value_type = list(value_type)
    path_inputs_dir = []
    path_niftis_in = []
    in_dir = utils.get_output_dir()
    work_id = (
        f"/{datetime.today().strftime('%Y%m%d-%H%M')}_"
        + f"{k_eigen}_{threshold}_{len(subjects_id)}_{clusters}_{batch_number}"
    )
    path_logs = (
        f"{utils.create_output_folder(in_dir, subjects_id[0], 'boostrap')}"
        + work_id
        + "_logs.txt"
    )
    path_nifti_out = [
        (
            f"{utils.create_output_folder(in_dir,subjects_id[0],'boostrap')}"
            + work_id
            + f"_{nifti_type}_{value}.nii.gz"
        )
        for value in value_type
    ]

    path_output_cluster = (
        f"{utils.create_output_folder(in_dir,subjects_id[0],'boostrap')}"
        + work_id
        + "_clusters.txt"
    )
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
        value_type,
        save,
    )
    # load the data
    logging.info("Loading data...")
    Us = []
    path_mask = utils.get_mask_path("95")
    wm_indices = nifti_utils.get_mask_ind(path_mask)

    for nifti in path_niftis_in:
        U, wm_indices = nifti_utils.extract_eigen_from_nifti(
            nifti, wm_indices, "population"
        )
        Us.append(U)

    # Produce the clustering matrix
    K = np.zeros((len(wm_indices), k_eigen * len(subjects_id)))
    for id in range(0, len(subjects_id)):
        U_subject = Us[id]
        for voxel in range(0, U_subject.shape[0]):
            start_col = id * k_eigen
            end_col = start_col + k_eigen
            K[voxel, start_col:end_col] = U_subject[voxel, 0:k_eigen]

    # Clean the clustering matrix
    nan_indices = np.where(np.isnan(K))[0]
    wm_indices = np.delete(wm_indices, nan_indices)
    K = np.delete(K, axis=0, obj=nan_indices)
    logging.info(f"Removed {len(nan_indices)} voxels")

    # Produce the kMean clustering
    K = whiten(K)
    centroids, dist = kmeans(K, clusters)
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
    for value, path_out in zip(value_type, path_nifti_out):
        nifti_utils.compute_nift(path_mask, path_output_cluster, path_out, value, save)


def create_annotation_file():
    """Method for creating the annotation files after computing all the boostrap"""
    Boostrap_path = (
        "/media/miplab-nas2/Data3/Hamid_Edouard/population_cluster/Bootstrap"
    )
    boostrap_list = os.listdir(Boostrap_path)
    dates = []
    n_patients = []
    n_eigens = []
    n_clusters = []
    n_batchs = []
    logs_paths = []
    # Extract general information
    for i in range(len(boostrap_list)):
        if boostrap_list[i][-8:] == "logs.txt":
            date, n_eigen, _, n_patient, n_cluster, n_batch, type = tuple(
                boostrap_list[i].split("_")
            )
            dates.append(date)
            n_patients.append(n_patient)
            n_eigens.append(n_eigen)
            n_clusters.append(n_cluster)
            n_batchs.append(n_batch)
            logs_paths.append(boostrap_list[i])

    df_annotation = pd.DataFrame.from_dict(
        {
            "date": dates,
            "n_patient": n_patients,
            "n_eigen": n_eigens,
            "n_cluster": n_clusters,
            "batch": n_batchs,
            "path_logs": logs_paths,
        }
    )

    # Check if all the files have been produced
    clusters_paths = []
    nifti_cluster_paths = []
    nifti_distance_paths = []
    nifti_z_paths = []
    for ind, data in df_annotation.iterrows():
        gen_str = (
            f"{data.date}_{data.n_eigen}_2.0_"
            + f"{data.n_patient}_{data.n_cluster}_{data.batch}"
        )
        path_cluster = f"{Boostrap_path}/{gen_str}_clusters.txt"
        path_nifit_cluster = f"{Boostrap_path}/{gen_str}_reslice_cluster.nii.gz"
        path_nifit_dist = f"{Boostrap_path}/{gen_str}_reslice_distance.nii.gz"
        path_nifit_z = f"{Boostrap_path}/{gen_str}_reslice_z_score.nii.gz"
        if (
            os.path.exists(path_cluster)
            & os.path.exists(path_nifit_cluster)
            & os.path.exists(path_nifit_dist)
            & os.path.exists(path_nifit_z)
        ):

            clusters_paths.append(path_cluster)
            nifti_cluster_paths.append(path_nifit_cluster)
            nifti_distance_paths.append(path_nifit_dist)
            nifti_z_paths.append(path_nifit_z)
        else:
            clusters_paths.append("None")
            nifti_cluster_paths.append("None")
            nifti_distance_paths.append("None")
            nifti_z_paths.append("None")

    df_annotation["path_cluster_txt"] = clusters_paths
    df_annotation["path_nifti_cluster"] = nifti_cluster_paths
    df_annotation["path_nifti_distance"] = nifti_distance_paths
    df_annotation["path_nifti_z_score"] = nifti_z_paths

    df_annotation = df_annotation.sort_values(["n_cluster", "batch"])
    df_annotation = df_annotation.reset_index(drop=True)
    df_annotation.to_csv(f"{Boostrap_path}/annotation.csv", index=False)
    return


if __name__ == "__main__":
    clustering_boostrap()
