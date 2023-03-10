import clustering.compute as compute
import clustering.utils as utils
import clustering.plot as plot
import matplotlib.pyplot as plt
import scipy.clustering.vq as kmeans


def clustering(patient_id, methods, thresold, k):
    # load the data
    A = utils.get_A(patient_id)
    # Preprocess the data
    A_wm = compute.compute_A_wm(A, patient_id)
    A_wm = compute.compute_fully_connected(A_wm)
    if thresold != 0:
        A_wm = compute.compute_binary_matrix(A_wm, thresold)
    # Compute all the required matrix
    D = compute.compute_D(A_wm)
    L = compute.compute_L(A_wm, D)

    # Compute the eigen vector
    W, V = compute.compute_eigenvalues(L, k)

    # Produce the kMean clustering
    V = kmeans.whiten(V)
    codebook, _ = kmeans.kmean(V, k)
    code = kmeans.vq(V, codebook)


return
