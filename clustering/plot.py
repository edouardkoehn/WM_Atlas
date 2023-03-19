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
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(np.arange(0, len(v)), v)
    ax.set_title(f"Eigen Values :{matrix_name}")
    ax.set_ylabel("Eigen values")
    ax.set_xlabel("k")
    ax.grid()
    plt.show()
