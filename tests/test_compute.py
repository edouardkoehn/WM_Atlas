import numpy as np
import pytest
from scipy import sparse

from clustering import compute


@pytest.mark.parametrize("A,ind_real,ind", [(np.ones((3, 3)), [10, 11, 12], [11])])
def test_remove_ind(A, ind_real, ind):
    A_filter, ind_filter = compute.remove_ind(A, ind_real, ind)
    assert A_filter.shape[0] == 1
    assert A_filter.shape[1] == 1
    assert ind_filter[0] == 11
    assert A_filter[0, 0] == 1


@pytest.mark.parametrize(
    "A,ind, output_size",
    [
        (np.matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]]), np.array([11, 23, 45]), 2),
        (
            np.matrix(
                [
                    [0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                ]
            ),
            np.array([1, 2, 3, 4, 5]),
            3,
        ),
    ],
)
def test_fully_connected(A, ind, output_size):
    out, ind = compute.compute_fully_connected(sparse.csr_matrix(A), ind)
    assert out.shape[0] == output_size
    assert out.shape[1] == output_size


@pytest.mark.parametrize(
    "A,s,ind", [(np.matrix([[0.5, 0.2], [0.8, 0.2]]), 0.5, np.array([11, 23, 45]))]
)
def test_compute_binary_matrix(A, s, ind):
    out, ind_out = compute.compute_binary_matrix(sparse.csr_matrix(A), s, ind)
    assert out.shape == A.shape
    assert np.sum(out[:, 0]) == 2
    assert np.sum(out[:, 1]) == 0
    assert len(ind_out) == len(ind)


@pytest.mark.parametrize(
    "A,ind", [(np.matrix([[0, 2, 5], [2, 0, 1], [5, 1, 0]]), np.array([11, 23, 45]))]
)
def test_compute_D(A, ind):
    out, ind_out = compute.compute_D(sparse.csr_matrix(A), ind)
    values = [7, 3, 6]
    for i in range(0, 3):
        assert out[i, i] == values[i]
    assert out[0, 1] == 0


@pytest.mark.parametrize(
    "A",
    [
        (np.matrix([[0, 2, 5], [2, 0, 1], [5, 1, 0]])),
        (
            np.matrix(
                [
                    [0, 1, 1, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 1, 1],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                ]
            )
        ),
    ],
)
def test_is_connected(A):
    assert compute.is_connected(sparse.csr_matrix(A))


@pytest.mark.parametrize("A", [(np.matrix([[0, 2, 5], [2, 0, 1], [5, 1, 0]]))])
def test_is_symetric(A):
    assert compute.is_symetric(sparse.csr_matrix(A))


def test_np_ravel():
    assert np.unravel_index(2, (3, 3), order="F") == (2, 0)
    assert np.unravel_index(7, (3, 3), order="F") == (1, 2)
    assert np.unravel_index(0, (3, 3, 3), order="F") == (0, 0, 0)
    assert np.unravel_index(17, (3, 3, 3), order="F") == (2, 2, 1)
