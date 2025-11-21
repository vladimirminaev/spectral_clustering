import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform


def get_distance_matrix_from_data(X, metric="euclidean"):
    """Return the pairwise distance matrix for the input samples.

    Parameters
    ----------
    X : ndarray
        Each row is a sample and each column is a feature.
    metric : str
        Distance metric name passed to ``pdist`` (e.g., 'euclidean').
        Default is ``"euclidean"``.

    Returns
    -------
    ndarray
        Square matrix of distances between every pair of rows in ``X``.
    """
    return squareform(pdist(X, metric=metric))


def get_adjacency_matrix_from_labels(labels):
    """Return a binary adjacency matrix where equal labels connect.

    Parameters
    ----------
    labels : array-like
        Label assigned to each sample. Entries with the same label receive
        adjacency value 1 (including self).

    Returns
    -------
    ndarray
        Boolean adjacency matrix cast to ints so matching labels produce 1s.
    """
    adjancency_matrix = np.equal.outer(labels, labels)
    adjancency_matrix = adjancency_matrix.astype(int)

    return adjancency_matrix


def get_similarity_matrix_from_distance_matrix(
    distance_matrix,
    sim_graph_type="fully_connect",
    sigma=1,
    knn=10,
    epsilon=0.5,
):
    """Return a similarity matrix constructed according to the selected graph.

    Parameters
    ----------
    distance_matrix : ndarray
        Pairwise distances between samples.
    sim_graph_type : str
        Similarity strategy: 'fully_connect' uses a Gaussian kernel, 'eps_neighbor'
        connects within ``epsilon`` distance, 'knn' links each point to its
        k nearest neighbors, and 'mutual_knn' keeps only mutual kNN links.
        Default is ``"fully_connect"``.
    sigma : float
        Bandwidth for the Gaussian kernel (used when ``sim_graph_type`` is
        'fully_connect'). Larger sigma softens the decay.
        Default is ``1``.
    knn : int
        Number of neighbors when building kNN or mutual kNN graphs.
        Default is ``10``.
    epsilon : float
        Distance threshold when building an epsilon-neighborhood graph.
        Default is ``0.5``.

    Returns
    -------
    ndarray
        Symmetric similarity matrix derived from the chosen graph rule.
    """
    # TODO: Think about self-edges. They do not change the Laplacian, but do we need to take care of them?
    if sim_graph_type == "fully_connect":
        W = np.exp(-distance_matrix / (2 * sigma))
    elif sim_graph_type == "eps_neighbor":
        W = (distance_matrix <= epsilon).astype("float64")
    elif sim_graph_type == "knn":
        W = np.zeros(distance_matrix.shape)

        # Sort the distance matrix by rows in ascending order and record the indices
        closest_neighbors = np.argsort(distance_matrix, axis=1)

        # Set the weight (i,j) to 1 when either i or j is within the k-nearest neighbors of each other
        for i in range(closest_neighbors.shape[0]):
            W[i, closest_neighbors[i, :][: (knn + 1)]] = 1
    elif sim_graph_type == "mutual_knn":
        W = np.zeros(distance_matrix.shape)

        # Sort the distance matrix by rows in ascending order and record the indices
        closest_neighbors = np.argsort(distance_matrix, axis=1)

        n = distance_matrix.shape[0]
        neighbors = np.zeros((n, n), dtype=bool)
        for i in range(n):
            neighbors[i, closest_neighbors[i]] = True

        # mutual k-NN: edge exists only when neighbor relation is mutual
        W = (neighbors & neighbors.T).astype("float64")
    else:
        raise ValueError(
            "The 'sim_graph_type' argument should be one of the strings, 'fully_connect', 'eps_neighbor', 'knn', or 'mutual_knn'!"
        )

    return W


def Spectral_Clustering(W, K=8, normalized=1, random_state=1):
    """Cluster the graph defined by ``W`` via spectral embedding + KMeans.

    Parameters
    ----------
    W : ndarray
        Similarity/weight matrix between ``n`` nodes (samples).
    K : int
        Number of clusters/eigenvectors to keep. Default ``8``.
    normalized : {0, 1, 2}
        1 = random-walk normalization (default), 2 = symmetric normalization,
        other values = unnormalized Laplacian.
    random_state : int
        Seed controlling KMeans initialization for reproducible results. Default
        ``1``.

    Returns
    -------
    tuple
        ``(kmeans, V_K, lambdas)`` with the fitted KMeans, selected eigenvectors,
        and their eigenvalues.
    """

    # Compute the unnormalized graph Laplacian
    D = np.diag(np.sum(W, axis=1))
    L = D - W

    # Random Walk normalized version
    if normalized == 1:
        D_inv = np.diag(1 / np.diag(D))
        eigenvalues, eigenvectors = np.linalg.eig(np.dot(D_inv, L))
        # Sort the eigenvalues by their L2 norms in ascending order and record the indices
        eigenvalue_indices_sorted = np.argsort(
            np.linalg.norm(np.reshape(eigenvalues, (1, len(eigenvalues))), axis=0)
        )
        V_K = np.real(eigenvectors[:, eigenvalue_indices_sorted[:K]])
    # Graph cut normalized version
    elif normalized == 2:
        D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(D)))
        eigenvalues, eigenvectors = np.linalg.eig(
            np.matmul(np.matmul(D_inv_sqrt, L), D_inv_sqrt)
        )
        # Sort the eigenvalues by their L2 norms in ascending order and record the indices
        eigenvalue_indices_sorted = np.argsort(
            np.linalg.norm(np.reshape(eigenvalues, (1, len(eigenvalues))), axis=0)
        )
        V_K = np.real(eigenvectors[:, eigenvalue_indices_sorted[:K]])
        if any(V_K.sum(axis=1) == 0):
            raise ValueError(
                "Can't normalize the matrix with the first K eigenvectors as columns! Perhaps the number of clusters K or the number of neighbors in k-NN is too small."
            )
        # Normalize the row sums to have norm 1
        V_K = V_K / np.reshape(np.linalg.norm(V_K, axis=1), (V_K.shape[0], 1))
    # Unnormalized version
    else:
        eigenvalues, eigenvectors = np.linalg.eig(L)
        # Sort the eigenvalues by their L2 norms in ascending order and record the indices
        eigenvalue_indices_sorted = np.argsort(
            np.linalg.norm(np.reshape(eigenvalues, (1, len(eigenvalues))), axis=0)
        )
        V_K = np.real(eigenvectors[:, eigenvalue_indices_sorted[:K]])

    kmeans = KMeans(n_clusters=K, init="k-means++", random_state=random_state).fit(V_K)

    # TODO: Do the eigenvalues of the normalized cases make any sense? Double check and is V_K the correct matrix to be returned
    # TODO: Epsilon threshold for the zero eigenvalues
    return (kmeans, V_K, eigenvalues[eigenvalue_indices_sorted[:K]])
