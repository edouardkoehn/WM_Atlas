import numpy as np
import scipy.sparse as sparse
import clustering.utils as utils
import networkx as nx


def remove_ind(A: sparse, ind_real: np.array, ind: np.array) -> sparse:
    """Return a sparse matrix containing only the indices specified in ind
    Arg:    A(sparse) sparse matrix
            ind_real: list of the real indices
            ind: list of the wanted real indices
    """
    spec_ind = np.isin(ind_real, ind)
    indices = [i for i, x in enumerate(spec_ind) if x]
    A_k = A[indices, :]
    A_k = A_k[:, indices]
    return A_k


def compute_fully_connected(A: sparse):
    """Check is an sparse matrix is fully connected"""
    connections = A.sum(axis=1).A1
    empty_nodes = np.where(connections != 0)[0]
    if len(empty_nodes) != 0:
        A_connected = A[empty_nodes, :]
        A_connected = A_connected[:, empty_nodes]
    else:
        A_connected = A
    return A_connected


def compute_A_wm(A: sparse, patient_id: int) -> sparse:
    """Return the Adjecent Matrix containing only the wm nodes"""
    real_ind = utils.get_ind(patient_id)
    wm_ind = utils.get_WM_ind(patient_id)
    return remove_ind(A, real_ind, wm_ind)


def compute_binary_matrix(A: sparse, thresold: float) -> sparse:
    """Methdo for producing a binary adjacent matrix"""
    A.data = np.where(A.data < thresold, 0, 1)
    return A


def compute_D(A: sparse) -> sparse:
    """Compute the diagonal matrix"""
    D = sparse.diags(A.sum(axis=1).A1)
    return D.tocsc()


def compute_L(A: sparse, D: sparse) -> sparse:
    """Compute the Laplacien"""
    return D - A


def compute_Lrw(A: sparse, D: sparse) -> sparse:
    D_inv = sparse.linalg.inv(D)
    L = compute_L(A, D)
    return D_inv - L


def compute_eigenvalues(L: sparse, k: int) -> sparse:
    """Compute the eigenvalues and eigenvecotrs of L"""
    eig_values, eig_vectors = sparse.linalg.eigs(L, k)
    return eig_values, eig_vectors


def is_connected(A: sparse) -> bool:
    """Check if the Sparse matrix is fully connected"""
    connections = A.sum(axis=1).A1
    empty_nodes = np.where(connections == 0)[0]
    if len(empty_nodes) == 0:
        return True
    else:
        return False
