import logging

import networkx as nx
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn import utils as sk

import clustering.utils as utils


def remove_ind(A: sparse, ind_real: np.array, ind: np.array) -> (sparse, list):
    """Return a sparse matrix containing only the indices specified in ind
    Arg:    A(sparse) sparse matrix
            ind_real: list of the real indices
            ind: list of the wanted real indices
    """
    spec_ind = np.isin(ind_real, ind)
    indices = [i for i, x in enumerate(spec_ind) if x]
    A_k = A[indices, :]
    A_k = A_k[:, indices]
    return (A_k, ind)


def compute_fully_connected(A: sparse, ind: list) -> (sparse, list):
    """Check is an sparse matrix is fully connected"""
    connections = A.sum(axis=1).A1
    full_nodes = np.where(connections != 0)[0]
    empty_nodes = np.where(connections == 0)[0]

    if not (len(empty_nodes) == 0):
        A_connected = A[full_nodes, :]
        A_connected = A_connected[:, full_nodes]
        ind = ind[full_nodes]
    else:
        A_connected = A
    return (A_connected, ind)


def compute_A_wm(A: sparse, patient_id: int) -> (sparse, list):
    """Return the Adjecent Matrix containing only the wm nodes"""
    real_ind = utils.get_ind(patient_id)
    wm_ind = utils.get_WM_ind(patient_id)
    return remove_ind(A, real_ind, wm_ind)


def compute_binary_matrix(A: sparse, thresold: float, ind: list) -> (sparse, list):
    """Methdo for producing a binary adjacent matrix"""
    A.data = np.where(A.data < thresold, 0, 1)
    return A, ind


def compute_D(A: sparse, ind: list) -> (sparse, list):
    """Compute the diagonal matrix"""
    D = sparse.diags(A.sum(axis=1).A1)
    return D.tocsc(), ind


def compute_L(A: sparse, ind: list) -> (sparse, list):
    """Compute the Laplacien"""
    D, _ = compute_D(A, ind)
    return D - A, ind


def compute_Lrw(A: sparse, ind: list) -> (sparse, list):
    logging.info(f"Starting to compute Lrw:")
    D_inv, _ = compute_D(A, ind)
    D_inv = sparse.linalg.inv(D_inv)
    logging.info(f"Finished to compute Lrw: {D_inv.shape, type(D_inv)}")
    L, _ = compute_L(A, ind)
    return D_inv * L, ind


def compute_eigenvalues(L: sparse, k: int, ind: list) -> (np.array, np.ndarray, list):
    """Compute the eigenvalues and eigenvecotrs of L"""
    logging.info(f"Starting to compute Eigen:")
    eig_values, eigen_vector = sparse.linalg.eigs(L, k=k, tol=5e-3, which="SM")
    logging.info(f"finieshed to compute eigen")

    return eig_values, eigen_vector, ind


def is_connected(A: sparse) -> bool:
    """Check if the Sparse matrix is fully connected"""
    connections = A.sum(axis=1).A1
    empty_nodes = np.where(connections == 0)[0]
    if len(empty_nodes) == 0:
        return True
    else:
        return False


def is_symetric(A: sparse) -> bool:
    try:
        sk.validation.check_symmetric(A)
    except:
        return False
    return True


def compute_nift(
    f_src_nifti_path: str,
    cluster_output_path: str,
    N_cluster: int,
    output_path: str,
    indices_raw: np.array,
):

    # load the nifti file
    h = nib.load(f_src_nifti_path)
    h.header[
        "descrip"
    ] = "Nifti files containing the cluster generated with spectral clustering (MIPLAB)"
    v = h.header["dim"][1:4]
    nifti_values = h.get_fdata()

    # load the cluster
    clusters = pd.read_csv(cluster_output_path)
    coord = np.zeros((clusters.shape[0], 3), dtype=int)
    for i in range(0, len(coord)):
        coord[i] = np.unravel_index(clusters["index"][i], (v[0], v[1], v[2]), order="F")
    clusters["x"] = coord[:, 0]
    clusters["y"] = coord[:, 1]
    clusters["z"] = coord[:, 2]

    clusters.C = clusters.C + 1
    # assigne the clusters
    nifti_values[clusters.x, clusters.y, clusters.z] = clusters.C
    # export the results
    output = nib.Nifti1Image(nifti_values, None, header=h.header.copy())
    nib.save(output, output_path)

    test = nib.load(output_path)
    print(test.get_fdata()[clusters.x[0], clusters.y[0], clusters.z[0]])
    print([clusters.x[0], clusters.y[0], clusters.z[0]])
    return
