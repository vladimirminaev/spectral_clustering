import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import plotly.graph_objects as go
import seaborn as sns
import math

PRESENTATION_COLORS = [
    "#8C1C13",  # deep burgundy
    "#F4D58D",  # muted gold
    "#A44A3F",  # warm brick
    "#6F5E76",  # desaturated violet
    "#F0A202",  # amber accent
    "#355070",  # slate blue for extras
]


class SpectralClusteringResult:
    """Encapsulate spectral clustering outputs and provide convenience accessors."""

    def __init__(self, clustering_model, eigenvectors, eigenvalues, labels, probs=None):
        self.clustering_model = clustering_model
        self.eigenvectors = np.asarray(eigenvectors)
        self.eigenvalues = np.asarray(eigenvalues)
        self.labels = np.asarray(labels)
        self.probs = None if probs is None else np.asarray(probs)


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
            hue=labels.astype(str),
            palette=palette_lookup,
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


def Spectral_Clustering(W, K=8, normalized=1, random_state=1, soft=False):
    """Cluster the graph defined by ``W`` via spectral embedding.

    Parameters
    ----------
    W : ndarray
        Similarity/weight matrix between ``n`` nodes.
    K : int
        Number of clusters/eigenvectors to keep.
    normalized : {0, 1, 2}
        1 = random-walk normalization (default), 2 = symmetric normalization,
        0 = unnormalized.
    random_state : int
        Seed for reproducibility.
    soft : bool, optional
        If True, uses Gaussian Mixture Models (GMM) for soft assignments.
        If False (default), uses KMeans for hard assignments.

    Returns
    -------
    SpectralClusteringResult
        Structured container with the fitted model, embedding ``V_K``, sorted
        eigenvalues, labels, and (optionally) posterior probabilities.
    """

    D = np.diag(np.sum(W, axis=1))
    L = D - W

    # Eigen decomposition based on normalization type
    if normalized == 1:
        D_inv = np.diag(1.0 / np.diag(D))
        target_matrix = np.dot(D_inv, L)

        eigenvalues, eigenvectors = np.linalg.eig(target_matrix)

    elif normalized == 2:
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        target_matrix = np.dot(np.dot(D_inv_sqrt, L), D_inv_sqrt)

        eigenvalues, eigenvectors = np.linalg.eig(target_matrix)

    else:
        target_matrix = L
        eigenvalues, eigenvectors = np.linalg.eig(target_matrix)

    eigenvalues = np.real(eigenvalues)
    eps = np.finfo(eigenvalues.dtype).eps
    eigenvalues[eigenvalues < 0] = eps
    eigenvectors = np.real(eigenvectors)

    indices = np.argsort(eigenvalues)
    eigenvalues_sorted = eigenvalues[indices]

    # Select first K eigenvectors
    V_K = eigenvectors[:, indices[:K]]

    if normalized == 2:
        # Normalize rows to unit norm
        row_norms = np.linalg.norm(V_K, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1e-10
        V_K = V_K / row_norms

    # Clustering
    if soft:
        gmm = GaussianMixture(
            n_components=K,
            covariance_type="full",
            random_state=random_state,
            n_init=10,
        )
        gmm.fit(V_K)
        labels = gmm.predict(V_K)
        probs = gmm.predict_proba(V_K)

        clustering_model = gmm
    else:
        kmeans = KMeans(n_clusters=K, init="k-means++", random_state=random_state)
        kmeans.fit(V_K)
        labels = kmeans.labels_
        probs = None
        clustering_model = kmeans

    return SpectralClusteringResult(
        clustering_model=clustering_model,
        eigenvectors=V_K,
        eigenvalues=eigenvalues_sorted,
        labels=labels,
        probs=probs,
    )


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

    eigs = np.asarray(np.real(eigenvalues), dtype=float).copy()
    eps = np.finfo(eigs.dtype).eps
    eigs[eigs < 0] = eps
    eigs_sorted = np.sort(np.abs(eigs))
    if eigs_sorted.size < 3:
        return 0.0

    # Skip the first eigenvalue gap (lambda_0 vs. lambda_1) as we assume >=2 clusters
    for idx in range(1, eigs_sorted.size - 1):
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
        return 2

    eigs = np.asarray(np.real(eigenvalues), dtype=float).copy()
    eps = np.finfo(eigs.dtype).eps
    eigs[eigs < 0] = eps
    eigs_sorted = np.sort(np.abs(eigs))

    if eigs_sorted.size <= 2:
        return 2

    # Skip the first eigenvalue gap (lambda_0 vs. lambda_1)
    for idx in range(1, eigs_sorted.size - 1):
        current_val = eigs_sorted[idx]
        next_val = eigs_sorted[idx + 1]

        if current_val < zero_tolerance and next_val >= spike_ratio * max(
            zero_tolerance, current_val
        ):
            spike = abs(next_val - current_val)
            if np.isclose(spike, gap, atol=match_tol):
                return idx + 1

    return 2


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
        eigs = np.asarray(eigs, dtype=float).copy()
        eps = np.finfo(eigs.dtype).eps
        eigs[eigs < 0] = eps
        k = min(n_first, len(eigs))
        vals = np.log(np.abs(eigs[:k]))
        spectra.append(vals)

    # Create one subplot per spectrum
    fig, axes = plt.subplots(
        1, n_graphs, figsize=(5.5 * n_graphs, 4.5), sharex=True, sharey=True
    )
    if n_graphs == 1:
        axes = [axes]

    for ax, vals, title in zip(axes, spectra, labels):
        x = np.arange(1, len(vals) + 1)
        ax.plot(
            x,
            vals,
            marker="o",
            markersize=6,
            linewidth=2,
            color=PRESENTATION_COLORS[0],
        )
        ax.set_xlabel("Eigenvalue index", fontsize=11)
        ax.set_ylabel("log(|eigenvalue|)", fontsize=11)
        ax.set_xticks(np.arange(1, n_first + 1))
        ax.set_xlim(1, n_first)
        ax.grid(False)
        ax.tick_params(axis="both", colors="#444444", labelsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Log First Eigenvalues", fontsize=14)
    fig.tight_layout(rect=[0, 0.01, 1, 0.92])


def plot_3d_spectral_embedding(
    spectral_embedding,
    true_labels=None,
    cluster_labels=None,
    add_jitter=True,
):
    """Interactive 3D embedding plot with cluster colors and optional true-label shapes."""

    if cluster_labels is None:
        raise ValueError("cluster_labels must be provided to color the embedding.")

    embedding = np.asarray(spectral_embedding)
    if embedding.shape[1] < 3:
        raise ValueError("spectral_embedding must provide at least three dimensions.")

    coords = embedding[:, :3].astype(float).copy()
    cluster_labels = np.asarray(cluster_labels)
    unique_clusters = np.unique(cluster_labels)

    label_array = None if true_labels is None else np.asarray(true_labels)
    unique_labels = [] if label_array is None else np.unique(label_array)

    base_palette = sns.color_palette(PRESENTATION_COLORS).as_hex()
    repeats = max(1, math.ceil(len(unique_clusters) / len(base_palette)))
    extended_palette = (base_palette * repeats)[: len(unique_clusters)]
    palette_lookup = {
        str(lbl): extended_palette[idx] for idx, lbl in enumerate(unique_clusters)
    }
    color_array = np.array(
        [palette_lookup[str(lbl)] for lbl in cluster_labels], dtype=object
    )
    cluster_custom = cluster_labels.astype(str).reshape(-1, 1)

    # scatter3d only supports a limited marker symbol set; stay within the valid list
    marker_cycle = [
        "circle",
        "circle-open",
        "square",
        "square-open",
        "diamond",
        "diamond-open",
        "cross",
        "x",
    ]
    marker_lookup = (
        {}
        if label_array is None
        else {
            str(lbl): marker_cycle[idx % len(marker_cycle)]
            for idx, lbl in enumerate(unique_labels)
        }
    )

    if add_jitter:
        spans = coords.max(axis=0) - coords.min(axis=0)
        spans[spans == 0] = 1.0
        rng = np.random.RandomState(42)
        coords += rng.normal(0, spans * 0.02, size=coords.shape)

    fig = go.Figure()

    if label_array is None:
        fig.add_trace(
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                name="Spectral embedding",
                marker=dict(
                    size=5,
                    color=color_array.tolist(),
                    line=dict(color="rgba(0,0,0,0.5)", width=0.4),
                    opacity=0.85,
                ),
                customdata=cluster_custom,
                hovertemplate=(
                    "Cluster label: %{customdata[0]}<br>"
                    "Eigenvector 1: %{x:.3f}<br>"
                    "Eigenvector 2: %{y:.3f}<br>"
                    "Eigenvector 3: %{z:.3f}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    else:
        for idx, lbl in enumerate(unique_labels):
            mask = label_array == lbl
            if not np.any(mask):
                continue
            fig.add_trace(
                go.Scatter3d(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    z=coords[mask, 2],
                    mode="markers",
                    name=f"True label {lbl}",
                    legendgroup="true_labels",
                    legendgrouptitle_text="True Label" if idx == 0 else None,
                    marker=dict(
                        size=5,
                        color=color_array[mask].tolist(),
                        symbol=marker_lookup[str(lbl)],
                        line=dict(color="rgba(0,0,0,0.5)", width=0.4),
                        opacity=0.85,
                    ),
                    customdata=np.column_stack(
                        [
                            cluster_labels[mask].astype(str),
                            label_array[mask].astype(str),
                        ]
                    ),
                    hovertemplate=(
                        "Cluster label: %{customdata[0]}<br>"
                        "True label: %{customdata[1]}<br>"
                        "Eigenvector 1: %{x:.3f}<br>"
                        "Eigenvector 2: %{y:.3f}<br>"
                        "Eigenvector 3: %{z:.3f}<extra></extra>"
                    ),
                )
            )

    for idx, cluster in enumerate(unique_clusters):
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                name=f"Cluster {cluster}",
                legendgroup="clusters",
                legendgrouptitle_text="Cluster Label" if idx == 0 else None,
                marker=dict(
                    size=6,
                    color=palette_lookup[str(cluster)],
                    symbol="circle",
                    line=dict(color="rgba(0,0,0,0.6)", width=0.6),
                ),
                visible="legendonly",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="Spectral Embedding (Interactive 3D)",
        scene=dict(
            xaxis_title="Eigenvector 1",
            yaxis_title="Eigenvector 2",
            zaxis_title="Eigenvector 3",
        ),
        legend=dict(
            orientation="v",
            x=1.02,
            y=1.0,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0,0,0,0.1)",
            itemsizing="constant",
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )

    fig.show()


def plot_2d_spectral_embedding(
    spectral_embedding, true_labels=None, cluster_labels=None, add_jitter=True
):
    """Render a 2D embedding plot that colors by clusters and (optionally) shapes by true labels."""

    sns.set_theme(style="white", context="talk")

    embedding = np.asarray(spectral_embedding)
    if cluster_labels is None:
        raise ValueError("cluster_labels must be provided to color the embedding.")

    cluster_labels = np.asarray(cluster_labels)
    unique_clusters = np.unique(cluster_labels)

    label_array = None if true_labels is None else np.asarray(true_labels)
    unique_labels = [] if label_array is None else np.unique(label_array)

    base_palette = sns.color_palette(PRESENTATION_COLORS)
    repeats = math.ceil(len(unique_clusters) / len(base_palette)) or 1
    extended_palette = (base_palette * repeats)[: len(unique_clusters)]
    palette_lookup = {
        str(lbl): extended_palette[idx] for idx, lbl in enumerate(unique_clusters)
    }
    color_array = np.array([palette_lookup[str(lbl)] for lbl in cluster_labels])

    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"]
    marker_lookup = (
        {}
        if label_array is None
        else {
            str(lbl): marker_cycle[idx % len(marker_cycle)]
            for idx, lbl in enumerate(unique_labels)
        }
    )

    plot_x = embedding[:, 0].copy()
    plot_y = embedding[:, 1].copy()

    if add_jitter:
        x_span = plot_x.max() - plot_x.min()
        y_span = plot_y.max() - plot_y.min()
        rng = np.random.RandomState(42)
        plot_x += rng.normal(0, x_span * 0.02, size=plot_x.shape)
        plot_y += rng.normal(0, y_span * 0.02, size=plot_y.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    if label_array is None:
        ax.scatter(
            plot_x,
            plot_y,
            c=color_array,
            marker="o",
            s=80,
            linewidths=0.6,
            edgecolor="black",
            alpha=0.9,
        )
    else:
        for lbl in unique_labels:
            mask = label_array == lbl
            if not np.any(mask):
                continue
            ax.scatter(
                plot_x[mask],
                plot_y[mask],
                c=color_array[mask],
                marker=marker_lookup[str(lbl)],
                s=80,
                linewidths=0.6,
                edgecolor="black",
                alpha=0.9,
            )

    ax.set_title("Spectral Embedding", pad=15)
    ax.set_xlabel("Eigenvector 1")
    ax.set_ylabel("Eigenvector 2")
    sns.despine(ax=ax, trim=False)
    ax.grid(True, linestyle=":", alpha=0.3)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=palette_lookup[str(cluster)],
            markeredgecolor="black",
            markersize=9,
        )
        for cluster in unique_clusters
    ]

    cluster_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_lookup[str(lbl)],
            color="black",
            linestyle="",
            markerfacecolor="none",
            markersize=8,
        )
        for lbl in unique_labels
    ]

    if legend_handles:
        cluster_legend = ax.legend(
            handles=legend_handles,
            labels=[str(cluster) for cluster in unique_clusters],
            title="Cluster Label",
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            frameon=False,
            borderpad=0.3,
            labelspacing=0.4,
        )
        cluster_legend._legend_box.align = "left"
        ax.add_artist(cluster_legend)

    if cluster_handles:
        true_legend = ax.legend(
            handles=cluster_handles,
            labels=[str(lbl) for lbl in unique_labels],
            title="True Label",
            loc="upper left",
            bbox_to_anchor=(1.02, 0.25),
            frameon=False,
            borderpad=0.3,
            labelspacing=0.4,
        )
        true_legend._legend_box.align = "left"

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    plt.show()


def diffusion_map(eigenvectors, eigenvalues, t=1):
    eigenvalues_diff = 1 - eigenvalues

    weighted_eigenvectors = eigenvectors * (eigenvalues_diff.reshape(1, -1) ** t)

    kmeans_diffmap = KMeans(n_clusters=eigenvectors.shape[1], random_state=1, n_init=10)
    y_pred_dmap = kmeans_diffmap.fit_predict(weighted_eigenvectors)

    return y_pred_dmap
