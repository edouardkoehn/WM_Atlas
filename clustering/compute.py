import networkx as nx
import numpy as np
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


def compute_L(A: sparse, D: sparse, ind: list) -> (sparse, list):
    """Compute the Laplacien"""
    return D - A, ind


def compute_Lrw(A: sparse, D: sparse, ind: list) -> (sparse, list):
    D_inv = sparse.linalg.inv(D)
    print("done")
    L, ind = compute_L(A, D, ind)
    return D_inv - L, ind


def compute_eigenvalues(L: sparse, k: int, ind: list) -> (np.array, np.ndarray, list):
    """Compute the eigenvalues and eigenvecotrs of L"""
    eig_values, eig_vectors = sparse.linalg.eigs(L, k)
    return eig_values, eig_vectors, ind


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
