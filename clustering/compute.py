import logging

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
    return (A_k, ind)


def compute_fully_connected(A: sparse, ind: np.array) -> tuple[sparse, np.array]:
    """Check is an sparse matrix is fully connected
    Args:   A(np.sparse): sparse Adjacent matrix
            ind(np.array): np.array of the corresponding indices of the matrix
            to the volume

    return  A_connected(np.sparse):sparse adjacent matrix fully connected
            ind (np.array):np.array of the corresponding indices of the matrix
            to the volume
    """
    connections = A.sum(axis=1).A1
    full_nodes = np.where(connections != 0)[0]
    empty_nodes = np.where(connections == 0)[0]
    if not (len(empty_nodes) == 0):
        A_connected = A[full_nodes, :]
        A_connected = A_connected[:, full_nodes]
        ind = ind[full_nodes]
    else:
        A_connected = A
    logging.info(f"A_wm fully connected, shape:{A_connected.shape}")
    return (A_connected, ind)


def compute_A_wm(A: sparse, patient_id: int) -> tuple[sparse, np.array]:
    """Return the Adjecent Matrix containing only the wm nodes
    Args:   A(np.sparse): Sparse raw adjacent matrix
            patient_id(int)
    """
    real_ind = utils.get_ind(patient_id)
    wm_ind = utils.get_WM_ind(patient_id)
    A_wm = remove_ind(A, real_ind, wm_ind)
    logging.info(f"A_wm, shape:{A.shape}")
    return A_wm


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
    L, _ = compute_L(A, ind)
    Lrw = D_inv * L
    if save:
        sparse.save_npz(path_matrix + "_Lrw.npz", Lrw)
    logging.info(f"Lrw shape:{L.shape}")
    return D_inv * L, ind


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
    eig_values, eigen_vector = sparse.linalg.eigs(L, k=k, tol=5e-3, which="SM")
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
    connections = A.sum(axis=1).A1
    empty_nodes = np.where(connections == 0)[0]
    if len(empty_nodes) == 0:
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
    path_nifit_in: str,
    cluster_path: str,
    path_nifti_out: str,
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
    h.header[
        "descrip"
    ] = "Nifti files containing the cluster generated with spectral clustering (MIPLAB)"
    v = h.header["dim"][1:4]
    nifti_values = h.get_fdata()

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
    # assigne the clusters
    nifti_values[clusters.x, clusters.y, clusters.z] = clusters.C
    # export the results
    output = nib.Nifti1Image(nifti_values, None, header=h.header.copy())
    nib.save(output, path_nifti_out)
    logging.info("Conversion finished")
    return
