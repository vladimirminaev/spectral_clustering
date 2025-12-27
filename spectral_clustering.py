import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import math
import pandas as pd

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

    def __init__(
        self,
        clustering_model,
        eigenvectors,
        eigenvalues,
        labels,
        probs=None,
        K=None,
        normalized=None,
        random_state=None,
        soft=None,
    ):
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
        K=K,
        normalized=normalized,
        random_state=random_state,
        soft=soft,
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

    # Create a compact grid so figures don't become ultra-wide (which forces
    # notebooks to downscale and makes each subplot tiny).
    if n_graphs <= 3:
        n_rows, n_cols = 1, n_graphs
    else:
        n_cols = 3
        n_rows = int(math.ceil(n_graphs / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.0 * n_cols, 3.9 * n_rows),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    axes = np.atleast_1d(axes).ravel()

    # For incomplete last rows (e.g., 5 plots on a 2x3 grid), the x-label should
    # appear on the bottom-most *existing* axis of each column.
    last_row_per_col = [0] * n_cols
    for col_idx in range(n_cols):
        if col_idx >= n_graphs:
            last_row_per_col[col_idx] = -1
            continue
        last_row_per_col[col_idx] = (n_graphs - 1 - col_idx) // n_cols

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= n_graphs:
            # Keep the bottom-most *empty* axis of a column visible so the
            # x-axis label also appears for columns that have no subplot in the
            # last row (e.g., 5 plots on a 2x3 grid).
            row_idx = ax_idx // n_cols
            col_idx = ax_idx % n_cols
            if row_idx == n_rows - 1 and last_row_per_col[col_idx] >= 0:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.tick_params(left=False, bottom=False)
                ax.set_ylabel("")
                ax.set_title("")
                ax.set_xlabel("Eigenvalue index", fontsize=12)
                ax.grid(False)
            else:
                ax.set_visible(False)
            continue

        vals = spectra[ax_idx]
        title = labels[ax_idx]
        x = np.arange(1, len(vals) + 1)
        ax.plot(
            x,
            vals,
            marker="o",
            markersize=6,
            linewidth=2,
            color=PRESENTATION_COLORS[0],
        )
        row_idx = ax_idx // n_cols
        col_idx = ax_idx % n_cols
        if n_rows == 1 or row_idx == last_row_per_col[col_idx]:
            ax.set_xlabel("Eigenvalue index", fontsize=12)
        if n_cols == 1 or col_idx == 0:
            ax.set_ylabel("log(|eigenvalue|)", fontsize=12)
        ax.set_xticks(np.arange(1, n_first + 1))
        ax.set_xlim(1, n_first)
        ax.grid(False)
        ax.tick_params(axis="both", colors="#444444", labelsize=12)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=14, color="#333333")

    fig.suptitle("Log First Eigenvalues", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0.02, 0.04, 1, 0.94])


def plot_3d_spectral_embedding(
    spectral_embedding,
    true_labels=None,
    cluster_labels=None,
    add_jitter=True,
    title="Spectral Embedding",
    figsize=(6.2, 4.6),
    elev=18,
    azim=35,
    marker_size=36,
    alpha=0.9,
    legend=True,
    savepath=None,
    dpi=300,
    show=True,
):
    """Slide-friendly static 3D embedding plot.

    Uses Matplotlib (not Plotly) to produce a crisp, exportable figure.

    Parameters
    ----------
    spectral_embedding : array-like
        Embedding coordinates with at least 3 columns.
    true_labels : array-like, optional
        Optional ground-truth labels. When provided, points are additionally
        distinguished by marker shape.
    cluster_labels : array-like
        Cluster assignments used for coloring.
    add_jitter : bool, default=True
        Adds a small deterministic jitter to reduce overplotting.
    title : str, default="Spectral Embedding"
        Figure title.
    figsize : tuple, default=(6.2, 4.6)
        Figure size in inches.
    elev, azim : float
        View angles for the 3D camera.
    marker_size : float
        Scatter marker area.
    alpha : float
        Marker opacity.
    legend : bool
        Whether to draw legends.
    savepath : str | None
        If provided, saves the figure to this path.
    dpi : int
        DPI for saving.
    show : bool
        If True, calls ``plt.show()``.

    Returns
    -------
    (fig, ax)
        Matplotlib figure and 3D axes.
    """

    if cluster_labels is None:
        raise ValueError("cluster_labels must be provided to color the embedding.")

    embedding = np.asarray(spectral_embedding)
    if embedding.ndim != 2 or embedding.shape[1] < 3:
        raise ValueError("spectral_embedding must be a 2D array with >= 3 columns.")

    coords = embedding[:, :3].astype(float).copy()
    cluster_labels = np.asarray(cluster_labels)
    if cluster_labels.shape[0] != coords.shape[0]:
        raise ValueError(
            "cluster_labels must have the same length as spectral_embedding."
        )

    unique_clusters = np.unique(cluster_labels)
    label_array = None if true_labels is None else np.asarray(true_labels)
    if label_array is not None and label_array.shape[0] != coords.shape[0]:
        raise ValueError("true_labels must have the same length as spectral_embedding.")

    if add_jitter:
        spans = coords.max(axis=0) - coords.min(axis=0)
        spans[spans == 0] = 1.0
        rng = np.random.RandomState(42)
        coords += rng.normal(0, spans * 0.02, size=coords.shape)

    base_palette = list(PRESENTATION_COLORS)
    repeats = max(1, math.ceil(len(unique_clusters) / len(base_palette)))
    extended_palette = (base_palette * repeats)[: len(unique_clusters)]
    palette_lookup = {
        cluster: extended_palette[i] for i, cluster in enumerate(unique_clusters)
    }

    # Improve contrast for the common case of clusters labeled 0/1/2:
    # cluster "2" can be too close to the burgundy "0", so assign a distinct
    # palette color (slate blue).
    if 2 in palette_lookup:
        palette_lookup[2] = PRESENTATION_COLORS[5]
    elif "2" in palette_lookup:
        palette_lookup["2"] = PRESENTATION_COLORS[5]

    if label_array is None:
        marker_lookup = None
        legend_true_handles = []
        legend_true_labels = []
    else:
        unique_labels = np.unique(label_array)
        marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"]
        marker_lookup = {
            lbl: marker_cycle[idx % len(marker_cycle)]
            for idx, lbl in enumerate(unique_labels)
        }
        legend_true_handles = [
            Line2D(
                [0],
                [0],
                marker=marker_lookup[lbl],
                color="black",
                linestyle="",
                markerfacecolor="none",
                markeredgecolor="black",
                markersize=6,
            )
            for lbl in unique_labels
        ]
        legend_true_labels = [str(lbl) for lbl in unique_labels]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Draw points (grouped for marker-shape control)
    if label_array is None:
        point_markers = ["o"]
        point_masks = [np.ones(coords.shape[0], dtype=bool)]
        point_labels = [None]
    else:
        point_markers = []
        point_masks = []
        point_labels = []
        for lbl, marker in marker_lookup.items():
            mask = label_array == lbl
            if np.any(mask):
                point_markers.append(marker)
                point_masks.append(mask)
                point_labels.append(lbl)

    for marker, mask in zip(point_markers, point_masks):
        colors = np.array(
            [palette_lookup[c] for c in cluster_labels[mask]], dtype=object
        )
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            coords[mask, 2],
            c=colors,
            s=marker_size,
            marker=marker,
            alpha=alpha,
            edgecolors="black",
            linewidths=0.35,
            depthshade=False,
        )

    title_fontsize = 11
    label_fontsize = 9
    tick_fontsize = 8
    text_color = "#333333"
    tick_color = "#444444"
    grid_alpha = 0.06

    # No title/axis label annotations (slide-friendly; add labels in the slide if desired).

    ax.tick_params(
        axis="both", which="major", labelsize=tick_fontsize, colors=tick_color
    )
    ax.view_init(elev=elev, azim=azim)

    # Clean, slide-friendly styling (match 2D: very light dotted grid)
    ax.grid(True)
    grid_rgba = (0.0, 0.0, 0.0, grid_alpha)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis._axinfo["grid"].update(
                {"color": grid_rgba, "linestyle": ":", "linewidth": 0.6}
            )
        except Exception:
            pass

    try:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((0.0, 0.0, 0.0, 0.0))
    except Exception:
        # Pane API can vary across Matplotlib versions; ignore if unsupported.
        pass

    if legend:
        # Legend: omit cluster-color legend entries (often redundant on slides).
        # If true labels are provided, keep the marker-shape legend below the plot.
        handles = legend_true_handles
        labels = legend_true_labels

        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=min(6, max(1, len(labels))),
                frameon=False,
                fontsize=tick_fontsize,
                bbox_to_anchor=(0.5, 0.0),
            )
            fig.tight_layout(rect=[0.0, 0.12, 1.0, 1.0])
        else:
            fig.tight_layout()
    else:
        fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_2d_spectral_embedding(
    spectral_embedding,
    true_labels=None,
    cluster_labels=None,
    add_jitter=True,
    title="Spectral Embedding",
    figsize=(6.2, 4.6),
):
    """Render a 2D embedding plot that colors by clusters and (optionally) shapes by true labels."""

    # Keep styling dependency-free (Matplotlib only).
    plt.rcParams.update(
        {
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    embedding = np.asarray(spectral_embedding)
    if cluster_labels is None:
        raise ValueError("cluster_labels must be provided to color the embedding.")

    cluster_labels = np.asarray(cluster_labels)
    unique_clusters = np.unique(cluster_labels)

    label_array = None if true_labels is None else np.asarray(true_labels)
    unique_labels = [] if label_array is None else np.unique(label_array)

    base_palette = list(PRESENTATION_COLORS)
    repeats = max(1, math.ceil(len(unique_clusters) / len(base_palette)))
    extended_palette = (base_palette * repeats)[: len(unique_clusters)]
    palette_lookup = {
        str(lbl): extended_palette[idx] for idx, lbl in enumerate(unique_clusters)
    }
    color_array = np.array(
        [palette_lookup[str(lbl)] for lbl in cluster_labels], dtype=object
    )

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

    fig, ax = plt.subplots(figsize=figsize)
    if label_array is None:
        ax.scatter(
            plot_x,
            plot_y,
            c=color_array,
            marker="o",
            s=55,
            linewidths=0.4,
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
                s=55,
                linewidths=0.4,
                edgecolor="black",
                alpha=0.9,
            )

    title_fontsize = 11
    label_fontsize = 9
    tick_fontsize = 8
    text_color = "#333333"
    tick_color = "#444444"
    grid_alpha = 0.0001

    # No title/axis label annotations (slide-friendly; add labels in the slide if desired).
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(
        True,
        linestyle=":",
        linewidth=0.6,
        color=(0.0, 0.0, 0.0, grid_alpha),
    )
    ax.tick_params(axis="both", labelsize=tick_fontsize, colors=tick_color)

    # Legend: omit cluster-color legend entries (often redundant on slides).
    # If true labels are provided, keep the marker-shape legend below the plot.
    true_handles = (
        [
            Line2D(
                [0],
                [0],
                marker=marker_lookup[str(lbl)],
                color="black",
                linestyle="",
                markerfacecolor="none",
                markersize=6,
            )
            for lbl in unique_labels
        ]
        if label_array is not None
        else []
    )

    labels = [str(lbl) for lbl in unique_labels] if label_array is not None else []
    if true_handles:
        fig.legend(
            true_handles,
            labels,
            loc="lower center",
            ncol=min(6, max(1, len(labels))),
            frameon=False,
            fontsize=tick_fontsize,
            bbox_to_anchor=(0.5, 0.0),
        )
        fig.tight_layout(rect=[0.0, 0.12, 1.0, 1.0])
    else:
        fig.tight_layout()

    plt.show()


def diffusion_map(eigenvectors, eigenvalues, t=1):
    eigenvalues_diff = 1 - eigenvalues

    weighted_eigenvectors = eigenvectors * (eigenvalues_diff.reshape(1, -1) ** t)

    kmeans_diffmap = KMeans(n_clusters=eigenvectors.shape[1], random_state=1, n_init=10)
    y_pred_dmap = kmeans_diffmap.fit_predict(weighted_eigenvectors)

    return y_pred_dmap


def plot_label_distribution(*label_sets, titles=None, palette=None, figsize=None):
    if not label_sets:
        raise ValueError("Provide at least one label array.")

    n_plots = len(label_sets)

    if titles is None:
        titles = [f"Distribution {idx + 1}" for idx in range(n_plots)]
    elif isinstance(titles, str):
        titles = [titles]
    if len(titles) != n_plots:
        raise ValueError("'titles' length must match the number of label arrays.")

    if palette is None:
        palette = PRESENTATION_COLORS
    if not isinstance(palette, (list, tuple)):
        palette = [palette]

    if figsize is None:
        figsize = (5.5 * n_plots, 4)

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    counts_list = []
    for idx, (labels, ax, title) in enumerate(zip(label_sets, axes, titles)):
        labels = np.asarray(labels)
        counts = (
            pd.Series(labels)
            .value_counts()
            .sort_index()
            .rename("count")
            .reset_index()
            .rename(columns={"index": "cluster"})
        )

        total = counts["count"].sum()
        total = total if total > 0 else 1
        counts["percentage"] = counts["count"] / total * 100.0

        color = palette[idx % len(palette)]
        ax.bar(
            counts["cluster"].astype(str),
            counts["percentage"],
            color=color,
            edgecolor="black",
        )
        ax.set_xlabel("Cluster label")
        ax.set_ylabel("Percentage")
        ax.set_ylim(0, 100)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle=":", alpha=0.3)
        counts_list.append(counts)

    fig.tight_layout()
    plt.show()

    return counts_list if n_plots > 1 else counts_list[0]
