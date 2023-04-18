import logging
import shutil
import subprocess

import nibabel as nib
import numpy as np
import pandas as pd

from clustering import utils


def compute_nift(
    path_nifit_in: str, cluster_path: str, path_nifti_out: str, save: bool
):
    """Method for converting the cluster into the nifti files
    Args:   path_nifti_in(str):path to the src nifti file
            cluster_path(str):path to the file containing
            the indices and their corresponding clusters
            N_clusters(int): number of cluster used
            path_nifti_in(str):path to the output nifti file
            indices_raw(np.array):np.array of the raw indices
    """
    logging.info("Converting the cluster into nifti ...")
    # load the nifti file
    h = nib.load(path_nifit_in)
    # nifti_values = (np.copy(h.get_fdata())).astype("int32")
    hd = h.header
    hd.set_data_dtype("float32")
    hd[
        "descrip"
    ] = "Nifti files containing the cluster generated with spectral clustering (MIPLAB)"
    v = h.header["dim"][1:4]
    nifti_values = np.zeros((v[0], v[1], v[2], 1))
    nifti_values[:] = np.nan

    # load the cluster
    clusters = pd.read_csv(cluster_path)
    indices = clusters["index"]
    # get the coordinate in x,y,z
    coord = np.zeros((indices.shape[0], 3), dtype=int)
    for i in range(0, len(coord)):
        coord[i] = np.unravel_index(indices[i], (v[0], v[1], v[2]), order="F")
    # Assigne the clusters
    for i in range(0, len(coord)):
        nifti_values[coord[i, 0], coord[i, 1], coord[i, 2], :] = clusters.C[i]
    # export the results
    output = nib.Nifti1Image(nifti_values, None, header=hd)
    if save:
        nib.save(output, path_nifti_out)
    logging.info("Conversion finished")
    return


def export_nift(
    path_nifit_in: str,
    U: np.array,
    ind: np.array,
    k: int,
    path_nifti_out: str,
    save: bool,
):
    """Method for exporting a nifty file that contains the
    eigenvectors value as pixel intensity
    Args:  path_nifti_in(str):path to the src nifti file
           U(np.array):Eigenvector matrix
           ind(np.array):list of the indices in the U matrix
           K_eigen: number of eigen value used for the extraction
           path_nifti_in(str):path to the output nifti file
           save(bool):
    """
    logging.info("Exporting the nifti ...")
    h = nib.load(path_nifit_in)
    hd = h.header
    hd.set_data_dtype("float32")
    hd["descrip"] = "Nifti files containing the eigenv val as pixel intensity"
    v = h.header["dim"][1:4]
    nifti_values = np.zeros((v[0], v[1], v[2], k))
    nifti_values[:] = np.nan

    # Get the coordinate in x,y,z
    coord = np.zeros((ind.shape[0], 3), dtype=int)
    for i in range(0, len(coord)):
        coord[i] = np.unravel_index(ind[i] - 1, (v[0], v[1], v[2]), order="F")
    # assigne the values
    for i in range(0, len(coord)):
        nifti_values[coord[i, 0], coord[i, 1], coord[i, 2], :] = U[i, :]

    output = nib.Nifti1Image(nifti_values, None, header=hd)
    if save:
        nib.save(output, path_nifti_out)
    logging.info(f"Nifti in the ACPC space exported: {path_nifti_out}")
    return


def copy_nifti(path_nifti_in: str, path_nifti_original_acpc_out: str, type: str, **f):
    """Method for copying the original nifti file into the ouput folder"""
    shutil.copyfile(path_nifti_in, path_nifti_original_acpc_out)
    subprocess.run(f"gzip {path_nifti_original_acpc_out}", shell=True, check=True)
    if type == "mni":
        hb_nii_displace(f["f_s"], f["f_d"], f["f_r"], f["f_o"])
    return


def extract_eigen_from_nifti(nifti_path: str, indices: np.array, type: str):
    """Extract the values of the nifti file, with the wm mask(flatten list of wm_ind)"""
    h = nib.load(nifti_path)
    data_3d = np.copy(h.get_fdata())
    data = np.reshape(
        data_3d,
        (data_3d.shape[0] * data_3d.shape[1] * data_3d.shape[2], 100),
        order="F",
    )

    if type == "individual":
        wm_ind = np.where(~np.isnan(data[:, 0]))[0]
        data = data[wm_ind, :]

    if type == "population":
        wm_ind = indices
        data = data[wm_ind, :]

    return data, wm_ind


def hb_nii_displace(f_s: str, f_d: str, f_r: str, f_o: str):
    """Method for converting the nifti file from the acpc2mni space
    Args:   f_s(str): Nifti file to convert to mni space
            f_d(std): Nifti file containing the transformation
            f_r(std): Nifti file containg the reference
            f_o(std): Nifti output file
    """
    root = utils.get_root() + "/matlab"
    spm12 = root + "/spm12"
    options = "-nodesktop -nodisplay -nosplash"
    command_init = f'addpath("{spm12}")'
    args = f'char("{f_s}"),char("{f_d}"),char("{f_r}"),char("{f_o}"),'
    extra_args = f'"{"InputFilesInReadOnlyDir"}", {"true"}'
    args = args + extra_args

    subprocess.run(
        f"matlab {options} -sd '{root}' -r '{command_init};"
        + f"hb_nii_displace({args});exit()' test.out > logs/log 2>&1 < stdio.txt ",
        shell=True,
        check=True,
    )
    logging.info(f"Nifti in the MNI space exported: {f_o}")

    return


def hb_reslice_vol(f_i: str, f_r: str, f_o):
    """Method for calling the hb_reslice_vol matlab function.
    Coregisteres an input volume to a given reference volume,
    and then reslices the volume so that the volume dimentions match,
    resulting in a one-to-one correspondence between the voxels of the output
    volume and the reference volume. The new resampled volume is written to
    the directory of the input volume, unless name of output file specified.
    """
    # Run the matlab function
    root = utils.get_root() + "/matlab"
    options = "-nodesktop -nodisplay -nosplash"
    command_init = f'addpath(genpath("{root}"))'
    args = f'char("{f_i}"),char("{f_r}"),[],char("{f_o}"),[],[0 1]'
    subprocess.run(
        f"matlab {options} -sd '{root}' -r '{command_init};"
        + f"hb_reslice_vol({args});exit()' test.out > logs/log 2>&1 < stdio.txt",
        shell=True,
        check=True,
    )
    logging.info(f"Nifti {f_i} resliced to  {f_o}")
    return


def get_mask_ind(path_mask: str) -> np.array:
    """Method for extracting the list of non-zerons indices in the mask"""
    h = nib.load(path_mask)
    data_3d = h.get_fdata()
    data = np.reshape(
        data_3d, (data_3d.shape[0] * data_3d.shape[1] * data_3d.shape[2], 1), order="F"
    )

    indices = np.where(data == 1.0)[0]
    return indices
