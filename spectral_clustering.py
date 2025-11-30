import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt


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
    mutual_knn=10,
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
        Number of neighbors when building kNN graphs.
        Default is ``10``.
    mutual_knn : int
        Number of neighbors when building mutual kNN graphs.
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

    # Define similarity matrix
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
            neighbors[i, closest_neighbors[i, :][: (mutual_knn + 1)]] = True

        # mutual k-NN: edge exists only when neighbor relation is mutual
        W = (neighbors & neighbors.T).astype("float64")
    else:
        raise ValueError(
            "The 'sim_graph_type' argument should be one of the strings, 'fully_connect', 'eps_neighbor', 'knn', or 'mutual_knn'!"
        )

    return W


def count_connected_components(W):
    """Return the number of connected components induced by ``W``.

    Parameters
    ----------
    W : ndarray
        Similarity (weight) matrix used to construct the adjacency graph.

    Returns
    -------
    int
        Count of connected components in the undirected graph defined by ``W``.
    """

    if W.size == 0:
        return 0

    adjacency = (W > 0).astype(int)
    graph = nx.from_numpy_array(adjacency)
    return nx.number_connected_components(graph)


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

    num_components = count_connected_components(W)
    if num_components > K:
        raise ValueError(
            "Similarity graph contains more connected components than the provided K."
        )

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
        eigenvalues_sorted = eigenvalues[eigenvalue_indices_sorted]

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
        eigenvalues_sorted = eigenvalues[eigenvalue_indices_sorted]

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
        eigenvalues_sorted = eigenvalues[eigenvalue_indices_sorted]

    kmeans = KMeans(n_clusters=K, init="k-means++", random_state=random_state).fit(V_K)

    # TODO: Do the eigenvalues of the normalized cases make any sense? Double check and is V_K the correct matrix to be returned
    # TODO: Epsilon threshold for the zero eigenvalues
    return (kmeans, V_K, eigenvalues_sorted)


def get_eigengap(eigenvalues, zero_tolerance=1e-3, spike_ratio=10.0):
    """Return the first eigengap spike following the provided heuristic.

    Parameters
    ----------
    eigenvalues : array-like
        Sequence of eigenvalues (real or complex) sorted internally before
        detecting spikes.
    zero_tolerance : float, default=1e-3
        Values with magnitude below this threshold count as "almost zero." The
        spike search begins from these near-zero eigenvalues.
    spike_ratio : float, default=10.0
        Required multiplicative jump from a near-zero eigenvalue to qualify as
        a spike. Must be positive.

    Returns
    -------
    float
        Magnitude of the first spike whose size is at least 10x larger than a
        near-zero eigenvalue (|lambda| < 1e-3). Returns ``0.0`` when no such
        spike exists.
    """

    if spike_ratio <= 0:
        raise ValueError("spike_ratio must be strictly positive")
    if zero_tolerance <= 0:
        raise ValueError("zero_tolerance must be strictly positive")

    eigs_sorted = np.sort(np.abs(np.real(eigenvalues)))
    if eigs_sorted.size < 2:
        return 0.0

    for idx in range(eigs_sorted.size - 1):
        current_val = eigs_sorted[idx]
        next_val = eigs_sorted[idx + 1]

        if current_val < zero_tolerance and next_val >= spike_ratio * max(
            zero_tolerance, current_val
        ):
            return float(abs(next_val - current_val))

    return 0.0


def estimate_k_from_eigengap(
    eigenvalues, zero_tolerance=1e-3, spike_ratio=10.0, match_tol=1e-6
):
    """Estimate the cluster count ``k`` via the eigengap heuristic.

    Parameters
    ----------
    eigenvalues : array-like
        Sequence of eigenvalues (typically from a Laplacian) used to locate
        the largest spectral gap.
    zero_tolerance : float, default=1e-3
        Passed through to ``get_eigengap`` so "almost-zero" eigenvalues are
        defined consistently when searching for the spike.
    spike_ratio : float, default=10.0
        Minimum multiplicative jump required between consecutive eigenvalues
        to declare an eigengap. Forwarded to ``get_eigengap``.
    match_tol : float, default=1e-6
        Absolute tolerance when matching the returned gap magnitude to the
        difference between consecutive eigenvalues.

    Returns
    -------
    int
        Index ``k`` such that the gap between ``lambda_k`` and ``lambda_{k+1}``
        matches the first spike detected by ``get_eigengap``. Returns ``0``
        when no spike is found.
    """

    if match_tol <= 0:
        raise ValueError("match_tol must be strictly positive")

    gap = get_eigengap(
        eigenvalues, zero_tolerance=zero_tolerance, spike_ratio=spike_ratio
    )
    if gap == 0.0:
        return 1

    eigs_sorted = np.sort(np.abs(np.real(eigenvalues)))

    if eigs_sorted.size == 1:
        return 1

    for idx in range(eigs_sorted.size - 1):
        current_val = eigs_sorted[idx]
        next_val = eigs_sorted[idx + 1]

        if current_val < zero_tolerance and next_val >= spike_ratio * max(
            zero_tolerance, current_val
        ):
            spike = abs(next_val - current_val)
            if np.isclose(spike, gap, atol=match_tol):
                return idx + 1

    return 1


def find_best_params_with_eigengap_grid_search(
    distance_matrix, param_grid, random_state=1
):
    """Return the parameter set that maximizes the eigengap.

    Parameters
    ----------
    distance_matrix : ndarray
        Pairwise distances between samples used to build similarity graphs.
    param_grid : dict or list of dicts
        Specification of parameter combinations compatible with ``ParameterGrid``.
    random_state : int, default=1
        Seed controlling the ``Spectral_Clustering`` call.

    Returns
    -------
    tuple
        ``(best_gap, best_params)`` with the largest eigengap and the parameter
        dictionary that produced it.
    """

    best_gap = -np.inf
    best_params = None

    for params in ParameterGrid(param_grid):
        similarity_matrix = get_similarity_matrix_from_distance_matrix(
            distance_matrix,
            sim_graph_type=params["sim_graph_type"],
            sigma=params["sigma"],
            knn=params["knn"],
            mutual_knn=params["mutual_knn"],
            epsilon=params["epsilon"],
        )

        try:
            _, _, eigenvalues = Spectral_Clustering(
                similarity_matrix,
                K=params["K"],
                random_state=random_state,
            )
        except ValueError as exc:
            if "connected components" in str(exc).lower():
                continue
            raise

        eigengap = get_eigengap(eigenvalues)

        if eigengap > best_gap:
            best_gap = eigengap
            best_params = params

            # We also try to estimate k if it does not raise an exemption. Otherwise, it is done manually.
            try:
                estimated_k = estimate_k_from_eigengap(eigenvalues)
                _, _, _ = Spectral_Clustering(
                    similarity_matrix,
                    K=estimated_k,
                    random_state=random_state,
                )
                best_params["K"] = estimated_k
            except ValueError as exc:
                if "connected components" in str(exc).lower():
                    continue
                raise

    return best_params


def plot_eigenvalues(eigenvalues_list, labels=None, n_first=10):
    """
    Plot log-spectra of the first `n_first` eigenvalues for a dynamic
    number of graphs.

    Parameters
    ----------
    eigenvalues_list : sequence of array-like
        List of arrays of eigenvalues to plot.
    labels : list of str, optional
        Labels for each eigenvalue array. If None, generic labels
        'Graph 1', 'Graph 2', ... are used. Length must match the
        number of provided eigenvalue arrays.
    n_first : int, default=10
        Number of leading eigenvalues to plot (starting from index 0).
    """

    if not eigenvalues_list:
        return

    n_graphs = len(eigenvalues_list)

    if labels is None:
        labels = [f"Graph {i + 1}" for i in range(n_graphs)]
    elif len(labels) != n_graphs:
        raise ValueError("Length of 'labels' must match number of eigenvalue arrays.")

    # Prepare log-eigenvalues, clipping to available eigenvalues
    spectra = []
    for eigs in eigenvalues_list:
        eigs = np.asarray(eigs, dtype=float)
        k = min(n_first, len(eigs))
        vals = np.log(np.abs(eigs[:k]))
        spectra.append(vals)

    # Create one subplot per spectrum
    fig, axes = plt.subplots(
        1, n_graphs, figsize=(4.5 * n_graphs, 4), sharex=True, sharey=True
    )
    if n_graphs == 1:
        axes = [axes]

    for ax, vals, title in zip(axes, spectra, labels):
        # x-axis starts from 1 instead of 0
        x = np.arange(1, len(vals) + 1)
        ax.plot(x, vals, marker="o")
        ax.set_title(title)
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("log(|eigenvalue|)")

    fig.suptitle("Log First Eigenvalues", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
