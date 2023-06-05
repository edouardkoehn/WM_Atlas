import logging

import clusim.sim as sim
import nibabel as nib
import numpy as np
import scipy.sparse as sparse
import scipy.spatial.distance as distance
from clusim.clustering import Clustering
from sklearn import utils as sk
from sklearn.metrics.cluster import normalized_mutual_info_score

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
    """Return the Adjacent Matrix containing only the wm nodes
    Args:   A(np.sparse): Sparse raw adjacent matrix
            patient_id(int): subject id


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
    """Method for producing a binary adjacent matrix
    Args:   A(np.sparse): Sparse raw adjacent matrix
            threshold(float): threshold value
            ind(np.array):: np.array of the corresponding indices
            of the matrix to the volume

    Returns     A(np.sparse)
                ind(np.array)
    """
    A.data = np.where(A.data < thresold, 0, 1)
    logging.info(f"A binarised:{A.shape}")
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
            ind(np.array):: np.array of the corresponding indices of
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


def compute_Lsym(
    A: sparse, ind: np.array, path_matrix: str, save: bool
) -> tuple[sparse, np.array]:
    """Compute the symetric Laplacien
    Args:   A(np.sparse): Sparse raw adjacent matrix
            ind(np.array):: np.array of the corresponding indices of
            the matrix to the volume
            path(str): path for saving the matrix
            save(bool):boolean for specified to save the matrix

    Returns:    L_sym(np.sparse)
                ind(np.array)
    """
    L_sym = sparse.csgraph.laplacian(A, normed=True)
    if save:
        sparse.save_npz(path_matrix + "_Lsym.npz", L_sym)
    logging.info(f"Lsym shape:{L_sym.shape}")
    return L_sym, ind


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
    L, _ = compute_L(A, ind, path_matrix, False)
    Lrw = D_inv.dot(L)
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
        which="SA",
        return_eigenvectors=True,
    )

    if save:
        np.save(path_matrix + "_U.npy", np.real(eigen_vector))
        np.save(path_matrix + "_v.npy", np.real(eig_values))
    logging.info(f"U, shape:{eigen_vector.shape}")
    return eig_values, eigen_vector, ind


def compute_sym_eigen_from_rw_eigen(U_rw: np.array, D: np.array):
    D_root = D.sqrt()
    U_sym = D_root.dot(U_rw)
    return U_sym


def compute_distance(centroids: np.array, features: np.array, assignements: np.array):
    """Method for computing the distance between each voxel and it's corresponding
    voxel
    Args:   centroids(np.array): Array(KxK) containing the centroids coordinates of
            each clusters
            features(np.array): Array(NXK) Features matrix
            assigmnement(np.array): Array(Nx1) Assignement matrix

    returns:    dist(np.array): Array(NX1) list of the distances"""
    dist = np.zeros(features.shape[0])

    for i in range(features.shape[0]):
        cluster = assignements[i]
        centroid = centroids[cluster]
        dist[i] = distance.euclidean(centroid, features[i, :])

    return dist


def compute_zscore(centroids: np.array, features: np.array, assignements: np.array):
    """Method for computing the z-score of each voxel"""
    dist = compute_distance(centroids, features, assignements)
    number_cluster = centroids.shape[0]

    # Produce the cluster matrix
    matrix_cluster = np.zeros((number_cluster, 2))
    for i in range(number_cluster):
        voxels_dist = dist[np.where(assignements == i)]
        matrix_cluster[i, 0] = np.mean(voxels_dist)
        matrix_cluster[i, 1] = np.std(voxels_dist)

    z_score = np.zeros(features.shape[0])
    for i in range(features.shape[0]):
        cluster_val = matrix_cluster[assignements[i]]
        z_score[i] = (dist[i] - cluster_val[0]) / cluster_val[1]

    return z_score


def compute_element_centric(
    path_vol_ref: str, path_vol_test: str, subdivision_shape=(5, 6, 5)
):
    """Method for computing the similarity between two clustered nifti files
    The two nifti files needed to be in the same space. This algorithm is using the
    centrimetric similarity metric.
    Here we are subdividing the space, and computing the similarity between
    each subspace.
    Args:   path_batch_ref(str)
            patch_batch_test(str)
            path_mask(str)
    """

    # Load the data
    vol_ref = np.zeros((260, 312, 260, 1))
    vol_test = np.zeros((260, 312, 260, 1))
    vol_ref[:, 0:-1, :] = nib.load(path_vol_ref).get_fdata()
    vol_test[:, 0:-1, :] = nib.load(path_vol_test).get_fdata()
    vol_ref[np.isnan(vol_ref)] = 0
    vol_test[np.isnan(vol_test)] = 0

    n_subdivision_x = int(vol_ref.shape[0] / subdivision_shape[0])
    n_subdivision_y = int(vol_ref.shape[1] / subdivision_shape[1])
    n_subdivision_z = int(vol_ref.shape[2] / subdivision_shape[2])

    dim_x_sub = subdivision_shape[0]
    dim_y_sub = subdivision_shape[1]
    dim_z_sub = subdivision_shape[2]

    distance_matrix = np.zeros((n_subdivision_x * n_subdivision_y * n_subdivision_z))
    subspace_ind = 0
    for z in range(n_subdivision_z):
        for y in range(n_subdivision_y):
            for x in range(n_subdivision_x):
                # Load the subpace
                C1 = vol_ref[
                    int(x * dim_x_sub) : int((x + 1) * dim_x_sub),
                    int(y * dim_y_sub) : int((y + 1) * dim_y_sub),
                    int(z * dim_z_sub) : int((z + 1) * dim_z_sub),
                ]
                C2 = vol_test[
                    int(x * dim_x_sub) : int((x + 1) * dim_x_sub),
                    int(y * dim_y_sub) : int((y + 1) * dim_y_sub),
                    int(z * dim_z_sub) : int((z + 1) * dim_z_sub),
                ]
                # Flaten the subset
                C1 = np.ravel(C1)
                C2 = np.ravel(C2)
                # Compute the similarity
                cluster1 = Clustering()
                cluster1.from_membership_list(C1)
                cluster2 = Clustering()
                cluster2.from_membership_list(C2)
                distance_matrix[subspace_ind] = sim.element_sim(
                    cluster1, cluster2, alpha=0.9
                )
                subspace_ind += 1

    return np.sum(distance_matrix) / (
        n_subdivision_x * n_subdivision_y * n_subdivision_z
    )


def compute_mni(path_vol_ref: str, path_vol_test: str):
    """Method for computing the similarity between two clustered nifti files
    The two nifti files need to be in the same space. This algorithm is using the NMI
    metric.
    Here we are computing the NMI metric on the complete space.
    Args:   path_batch_ref(str)
            patch_batch_test(str)
    """
    # Load the data
    vol_ref = np.zeros((260, 312, 260, 1))
    vol_test = np.zeros((260, 312, 260, 1))
    vol_ref[:, 0:-1, :] = nib.load(path_vol_ref).get_fdata()
    vol_test[:, 0:-1, :] = nib.load(path_vol_test).get_fdata()
    vol_ref[np.isnan(vol_ref)] = 0
    vol_test[np.isnan(vol_test)] = 0

    # Flaten the subset
    C1 = np.ravel(vol_ref)
    C2 = np.ravel(vol_test)
    # Compute the similarity
    dist = normalized_mutual_info_score(C1, C2)
    return dist


def compute_simlarity(experiments_list: list, metric: str = "MNI"):
    """Methods for computing the similarity matrix on a list of clustering
    The Similarity matrix is symetric where v[i,j] is equal to the similarity between
    the cluster i and the the j in the experiments_list
    Args:   experiments_list(list(str)): list containing the path of the clustering
            files
            metrics(str): which metric is used for computing the similarity (NMI,
            Centrimetric)"""

    columns = experiments_list
    rows = experiments_list
    distance_matrix = np.zeros((len(rows), len(columns)))
    # Build the distance matrix
    for row in range(len(rows)):
        for line in range(row, len(columns)):
            if metric == "MNI":
                distance_matrix[row, line] = compute_mni(rows[row], columns[line])
                distance_matrix[line, row] = distance_matrix[row, line]
            elif metric == "Centric":
                distance_matrix[row, line] = compute_element_centric(
                    rows[row], columns[line]
                )
                distance_matrix[line, row] = distance_matrix[row, line]
    return distance_matrix


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


def fiedler_vector(v: list):
    """Method for finding the fielder vector and return the fielder indices and the
    constant vector that is right before the fiedler vector
    The fielder is the first eigen value that is greater than e-5"""
    idx = (np.where(v > 1e-5))[0][0]
    return idx, idx + 1


def jaccard_similarity(list1, list2):
    """
    Calculates the Jaccard similarity between two lists.

    Parameters:
    list1 (list): The first list to compare.
    list2 (list): The second list to compare.
    if equal to one --> both ensmble as equal
    Returns:
    float: The Jaccard similarity between the two lists.
    """
    # Convert the lists to sets for easier comparison
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))
