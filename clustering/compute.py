import numpy as np
import scipy.sparse as sparse
import clustering.utils as utils


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


A = utils.get_A(100307)
ind_wm = utils.get_WM_ind(100307)
ind_real = utils.get_ind(100307)
print(ind_wm.shape)
print(ind_real.shape)
print(A.shape)
print(type(A))
print(remove_ind(A, ind_real, ind_wm).shape)
