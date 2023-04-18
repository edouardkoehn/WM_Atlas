import logging

import numpy as np
import scipy.sparse as sparse
from sklearn import utils as sk

import clustering.utils as utils


def remove_ind(A: sparse, ind_real: np.array, ind: np.array) -> tuple[sparse, np.array]:
    """Return a sparse matrix containing only the indices specified in ind
    Arg:    A(sparse) sparse matrix
            ind_real: np.array of the real indices
            ind: np.array of the wanted real indices

    Returns  A_k(sparse):sparse matrix containing only the corresponding index of ind
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

    Returns  A_connected(np.sparse):sparse adjacent matrix fully connected
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


    Returns     A_wm(np.array)
                ind(np.array)
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


    Returns     A(np.sparse)
                ind(np.array)
    """
    A.data = np.where(A.data < thresold, 0, 1)
    return A, ind


def compute_D(A: sparse, ind: np.array) -> tuple[sparse, np.array]:
    """Compute the diagonal matrix
    Args:   A(np.sparse): Sparse raw adjacent matrix
            ind(np.array):: np.array of the corresponding indices
            of the matrix to the volume


    Returns     D(np.array)
                ind(np.array)
    """
    D = sparse.diags(A.sum(axis=1).A1)
    return D.tocsc(), ind


def compute_L(
    A: sparse, ind: np.array, path_matrix: str, save: bool
) -> tuple[sparse, np.array]:
    """Compute the combinatorial Laplacien
    Args:   A(np.sparse): Sparse raw adjacent matrix
            ind(np.arry):: np.array of the corresponding indices of
            the matrix to the volume
            path(str): path for saving the matrix
            save(bool):boolean for specified to save the matrix

    Returns:    L(np.sparse)
                ind(np.array)
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
    """Compute the random walk Laplacien
    Args:   A(np.sparse): Sparse raw adjacent matrix
            ind(np.array):: np.array of the corresponding indices of the matrix
            to the volume
            path(str): path for saving the matrix
            save(bool): boolean for specified to save the matrix

    Returns:    Lrw(np.sparse)
                ind(np.array)
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
    Args:   L(np.sparse): Sparse raw adjacent matrix
            k(int):number of eigen value to compute
            ind(np.array): np.array of the corresponding indices of the matrix
            to the volume
            path_matrix(str): path for saving the matrix
            save(bool):boolean for specifing if the matrix needs to be saved

    Returns:    eigen_values(np.array)
                eigen_vector(np.array)
                ind(np.array)

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

    return  True if fully connected
    """
    n_connected, label = sparse.csgraph.connected_components(A, directed=False)
    if n_connected == 1:
        return True
    else:
        return False


def is_symetric(A: sparse) -> bool:
    """Method for checking if a matrix if symetrix
    Args:   A(np.sparse): Sparse matrix to check

    retun   True if symetric
    """
    A_sym = sk.validation.check_symmetric(A)
    if np.sum(A_sym != A) == 0:
        return True
    else:
        return False
