from datetime import datetime

import click
import numpy as np

import clustering.compute as compute
import clustering.utils as utils


@click.command()
@click.option("-i", "--subject_id", type=int, required=True, help="""Patient id""")
@click.option(
    "-m",
    "--method",
    type=click.Choice(["comb", "rw"], case_sensitive=False),
    required=False,
    default="acpc",
    help="""Clustering method used (comb:Combinatorial(default),
    rd:Random Walk version)""",
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
    required=False,
    default="acpc",
    help="""Type of nifti saved""",
)
@click.option(
    "-s",
    "--save",
    type=bool,
    required=False,
    default=False,
    help="""Saving the intermediate matrix (L, Lrw)""",
)
def extract_eigen(
    subject_id: int,
    method: str = "comb",
    threshold: float = 2,
    k_eigen: int = 10,
    nifti_type: str = "acpc",
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
    path_output_dir = utils.create_output_folder(
        utils.get_output_dir(), subject_id, "subject"
    )
    work_id = f"/{datetime.today().strftime('%Y%m%d-%H%M')}_{subject_id}_{threshold}"
    path_logs = path_output_dir + work_id + "_extraction_logs.txt"
    path_nifti_in = utils.load_data(subject_id)["G"]["f"]["mask"]
    path_nifti_out = path_output_dir + work_id + "_extraction_acpc.nii.gz"
    path_nifti_original_acpc_out = path_output_dir + work_id + "_original_acpc.nii.gz"
    path_matrix = path_output_dir + work_id

    if nifti_type == "mni":
        path_nifti_original_mni_out = path_output_dir + work_id + "_original_mni.nii.gz"
        f_s = path_nifti_out
        f_d = utils.get_transformation_file(subject_id, "acpc2nmi")
        f_r = utils.get_reference_file(subject_id)
        f_o = path_output_dir + work_id + "_extraction_mni.nii.gz"

    utils.create_logs(
        path_logs,
        subject_id,
        datetime.today().strftime("%Y%m%d_%H:%M"),
        method,
        threshold,
        k_eigen,
        nifti_type,
        save,
    )
    # load the data
    A = utils.get_A(subject_id)

    # Preprocess the data
    A_wm, ind = compute.compute_A_wm(A, subject_id)

    if threshold != 2:
        A_wm, ind = compute.compute_binary_matrix(A_wm, threshold, ind)

    A_wm, ind = compute.compute_fully_connected(A_wm, ind)
    # A_wm=A_wm[0:100,0:100]
    # ind=ind[0:100]
    # Compute all the required matrix
    if method == "comb":
        L, ind = compute.compute_L(A_wm, ind, path_matrix, save)
    if method == "rw":
        L, ind = compute.compute_Lrw(A_wm, ind, path_matrix, save)

    # Compute the eigen vector
    v, U, ind = compute.compute_eigenvalues(L, k_eigen, ind, path_matrix, save)
    v = np.abs(v)

    # Export the results
    compute.export_nift(path_nifti_in, U, ind, k_eigen, path_nifti_out, save)

    # Convert the results to nifti in the acpc space
    if nifti_type == "mni":
        compute.hb_nii_displace(f_s, f_d, f_r, f_o)

    # Copy the original nifti
    if nifti_type == "mni":
        compute.copy_nifti(
            path_nifti_in,
            path_nifti_original_acpc_out,
            type=nifti_type,
            f_s=path_nifti_in,
            f_d=f_d,
            f_r=f_r,
            f_o=path_nifti_original_mni_out,
        )
    elif nifti_type == "acpc":
        compute.copy_nifti(path_nifti_in, path_nifti_original_acpc_out, type=nifti_type)


if __name__ == "__main__":
    extract_eigen()
