import logging
import shutil
import subprocess

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn import utils as sk

import clustering.utils as utils


def remove_ind(A: sparse, ind_real: np.array, ind: np.array) -> tuple[sparse, np.array]:
    """Return a sparse matrix containing only the indices specified in ind
    Arg:    A(sparse) sparse matrix
            ind_real: np.array of the real indices
            ind: np.array of the wanted real indices

    return  A_k(sparse):sparse matrix containing only the corresponding index of ind
            ind: np.array of the corresponding indices of the matrix to the volume
    """
    spec_ind = np.isin(ind_real, ind)
    indices = [i for i, x in enumerate(spec_ind) if x]
    A_k = A[indices, :]
    A_k = A_k[:, indices]
    return A_k, ind


def compute_fully_connected(A: sparse, ind: np.array) -> tuple[sparse, np.array]:
    """Check is an sparse matrix is fully connected
    Args:   A(np.sparse): sparse Adjacent matrix
            ind(np.array): np.array of the corresponding indices of the matrix
            to the volume

    return  A_connected(np.sparse):sparse adjacent matrix fully connected
            ind (np.array):np.array of the corresponding indices of the matrix
            to the volume
    """
    n_connections, labels = sparse.csgraph.connected_components(A, directed=False)

    if n_connections != 1:
        masks = []
        for n in range(0, n_connections):
            masks.append(np.where(labels == n, True, False))
        size_mask = [np.sum(i) for i in masks]
        index_biggest_mask = size_mask.index(np.max(size_mask))

        mask = masks[index_biggest_mask]
        A_fully = A[mask, :]
        A_fully = A_fully[:, mask]
        ind_fully = ind[mask]

    else:
        A_fully = A
        ind_fully = ind

    logging.info(f"A_wm fully connected, shape:{A_fully.shape}")
    return A_fully, ind_fully


def compute_A_wm(A: sparse, patient_id: int) -> tuple[sparse, np.array]:
    """Return the Adjecent Matrix containing only the wm nodes
    Args:   A(np.sparse): Sparse raw adjacent matrix
            patient_id(int)
    """
    real_ind = utils.get_ind(patient_id)
    wm_ind = utils.get_WM_ind(patient_id)
    A_wm, ind = remove_ind(A, real_ind, wm_ind)
    logging.info(f"A_wm, shape:{A_wm.shape}")
    return A_wm, ind


def compute_binary_matrix(
    A: sparse, thresold: float, ind: np.array
) -> tuple[sparse, np.array]:
    """Methdo for producing a binary adjacent matrix
    Args:   A(np.sparse): Sparse raw adjacent matrix
            threshold(float): threshold values
            ind(np.array):: np.array of the corresponding indices
            of the matrix to the volume
    """
    A.data = np.where(A.data < thresold, 0, 1)
    return A, ind


def compute_D(A: sparse, ind: np.array) -> tuple[sparse, np.array]:
    """Compute the diagonal matrix
    Args:   A(np.sparse): Sparse raw adjacent matrix
            ind(np.array):: np.array of the corresponding indices
            of the matrix to the volume
    """
    D = sparse.diags(A.sum(axis=1).A1)
    return D.tocsc(), ind


def compute_L(
    A: sparse, ind: np.array, path_matrix: str, save: bool
) -> tuple[sparse, np.array]:
    """Compute the Laplacien
    Args:   A(np.sparse): Sparse raw adjacent matrix
            ind(np.arry):: np.array of the corresponding indices of
            the matrix to the volume
    """
    D, _ = compute_D(A, ind)
    L = D - A
    if save:
        sparse.save_npz(path_matrix + "_L.npz", L)
    logging.info(f"L shape:{L.shape}")
    return L, ind


def compute_Lrw(
    A: sparse, ind: np.array, path_matrix: str, save: bool
) -> tuple[sparse, np.array]:
    """Compute the noramlized Laplacien
    Args:   A(np.sparse): Sparse raw adjacent matrix
            ind(np.array):: np.array of the corresponding indices of the matrix
            to the volume
    """
    logging.info("Starting to compute Lrw:")
    D_inv, _ = compute_D(A, ind)
    D_inv = sparse.linalg.inv(D_inv)
    logging.info(f"Finished to compute Lrw: {D_inv.shape, type(D_inv)}")
    L, _ = compute_L(A, ind, path_matrix, False)
    Lrw = D_inv * L
    if save:
        sparse.save_npz(path_matrix + "_Lrw.npz", Lrw)
    logging.info(f"Lrw shape:{Lrw.shape}")
    return Lrw, ind


def compute_eigenvalues(
    L: sparse, k: int, ind: np.array, path_matrix: str, save: bool
) -> tuple[np.array, np.array, np.array]:
    """Compute the eigenvalues and eigenvecotrs of L
    Args:   A(np.sparse): Sparse raw adjacent matrix
            k(int):number of eigen value to compute
            ind(np.array):: np.array of the corresponding indices of the matrix
            to the volume
    """
    logging.info("Computing the eigen values...")
    eig_values, eigen_vector = sparse.linalg.eigsh(
        L,
        k=k,
        tol=1e-1,
        which="SA",
        v0=np.ones(L.shape[0]) * 0.01,
        return_eigenvectors=True,
    )
    if save:
        np.save(path_matrix + "_U.npy", np.real(eigen_vector))
        np.save(path_matrix + "_v.npy", np.real(eig_values))
    logging.info(f"U, shape:{eigen_vector.shape}")
    return eig_values, eigen_vector, ind


def is_connected(A: sparse) -> bool:
    """Check if the Sparse matrix is fully connected
    Args:   A(np.sparse): Sparse raw adjacent matrix

    return  bool
    """
    n_connected, label = sparse.csgraph.connected_components(A, directed=False)
    if n_connected == 1:
        return True
    else:
        return False


def is_symetric(A: sparse) -> bool:
    """Method for checking if a matrix if symetrix"""
    A_sym = sk.validation.check_symmetric(A)
    if np.sum(A_sym != A) == 0:
        return True
    else:
        return False


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
    hd["data_type"] = "int32"
    hd[
        "descrip"
    ] = "Nifti files containing the cluster generated with spectral clustering (MIPLAB)"
    v = h.header["dim"][1:4]
    nifti_values = np.zeros((v[0], v[1], v[2], 1)).astype(int)
    # load the cluster
    clusters = pd.read_csv(cluster_path)
    coord = np.zeros((clusters.shape[0], 3), dtype=int)
    for i in range(0, len(coord)):
        coord[i] = np.unravel_index(
            clusters["index"][i] + 1, (v[0], v[1], v[2]), order="F"
        )

    clusters["x"] = coord[:, 0]
    clusters["y"] = coord[:, 1]
    clusters["z"] = coord[:, 2]
    clusters.C = clusters.C + 1
    print(clusters)
    # assigne the clusters
    print(nifti_values[clusters.x[0], clusters.y[0], clusters.z[0], :])
    print(clusters.C[0])
    nifti_values[clusters.x, clusters.y, clusters.z, :] = np.array(clusters.C).reshape(
        (len(clusters.x), 1)
    )

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
    hd["data_type"] = "float32"
    hd["descrip"] = "Nifti files containing the eigenv val as pixel intensity"
    v = h.header["dim"][1:4]
    nifti_values = np.zeros((v[0], v[1], v[2], k))

    # Get the coordinate in x,y,z
    coord = np.zeros((ind.shape[0], 3), dtype=int)
    for i in range(0, len(coord)):
        coord[i] = np.unravel_index(ind[i] + 1, (v[0], v[1], v[2]), order="F")

    for i in range(0, len(U)):
        nifti_values[coord[i, 0], coord[i, 1], coord[i, 2], :] = U[i, :]

    output = nib.Nifti1Image(nifti_values, None, header=hd)

    if save:
        nib.save(output, path_nifti_out)
    logging.info(f"Nifti in the ACPC space exported: {path_nifti_out}")
    return


def copy_nifti(path_nifti_in: str, path_nifti_original_acpc_out: str, type: str, **f):
    """Method for copying the original nifti file into the ouput folder"""
    if type == "mni":
        # shutil.copyfile(f['f_s'],f['f_o'])
        hb_nii_displace(f["f_s"], f["f_d"], f["f_r"], f["f_o"])
        shutil.copyfile(path_nifti_in, path_nifti_original_acpc_out)

    return


def extract_eigen_from_nifti(nifti_path: str, indices: np.array):
    """Extract the values of the nifti file, with the wm mask(flatten list of wm_ind)"""
    h = nib.load(nifti_path)
    data = np.copy(h.get_fdata())
    unpacked_ind = np.unravel_index(
        indices + 1, (data.shape[0], data.shape[1], data.shape[2]), order="F"
    )

    data = data[unpacked_ind]
    return data


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
        + f"hb_nii_displace({args});exit()'",
        shell=True,
        check=True,
    )
    logging.info(f"Nifti in the MNI space exported: {f_o}")
    return
