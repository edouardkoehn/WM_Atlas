import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse


def plot_connections_distribution(A: sparse, matrix_name=""):
    """Plot the number of connection per nodes"""
    connection = A.sum(axis=1).A1
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(connection, bins=np.arange(0, np.max(connection)))

    ax.set_title(f"Connections distribution :{matrix_name}")
    ax.set_ylabel("# nodes")
    ax.set_xlabel("# connection")
    plt.show()


def plot_nodes_eigen(v: sparse, matrix_name=""):
    """Method for plotting the eignvalues repartitions"""
    v = np.sort(v)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(np.arange(0, v.shape[0]), v)
    ax.set_title(f"Eigen Values :{matrix_name}")
    ax.set_ylabel("Eigen values")
    ax.set_xlabel("k")
    # ax.set_xlim([0,10])
    # ax.set_ylim([0,0.004])
    ax.grid()
    plt.show()
