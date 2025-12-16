import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
from spectral_clustering import *

from spectral_clustering import (
    Spectral_Clustering,
    get_distance_matrix_from_data,
    get_similarity_matrix_from_distance_matrix,
)


def match_labels(y_true, y_pred):
    """Match predicted labels to ground truth via the Hungarian algorithm.

    Parameters
    ----------
    y_true : array-like
        Ground-truth cluster assignments.
    y_pred : array-like
        Raw labels returned by a clustering algorithm.

    Returns
    -------
    tuple of (ndarray, ndarray)
        ``(cm_matched, y_pred_matched)`` where ``cm_matched`` is the reordered
        confusion matrix and ``y_pred_matched`` contains the relabeled
        predictions aligned to ``y_true``.
    """

    cm = confusion_matrix(y_true, y_pred)

    # row_ind will be [0, 1, 2...] (the original indices)
    # col_ind will be the corresponding best matching index in y_pred
    row_ind, col_ind = linear_sum_assignment(cm, maximize=True)

    # Create a dictionary to map old predicted labels to new aligned labels
    # If the algorithm says Row 0 matches Column 2, we map label 2 -> 0.
    map_dict = {old_label: new_label for new_label, old_label in zip(row_ind, col_ind)}

    # Remap the predicted labels
    y_pred_matched = np.vectorize(map_dict.get)(y_pred)

    # Compute the new, "Diagonalized" Confusion Matrix
    cm_matched = cm[row_ind, :][:, col_ind]

    return cm_matched, y_pred_matched


def perturb_data(X, noise_std, rng=None):
    """Return a perturbed copy of ``X`` by adding Gaussian noise.

    Parameters
    ----------
    X : ndarray
        Input data matrix whose rows are samples.
    noise_std : float
        Standard deviation of the isotropic Gaussian noise.
    rng : np.random.RandomState, optional
        Source of randomness; falls back to ``np.random.RandomState`` seeded by
        ``random_state``.

    Returns
    -------
    ndarray
        Noisy data with the same shape as ``X``.
    """
    if rng is None:
        rng = np.random.RandomState(random_state)
    return X + rng.normal(loc=0.0, scale=noise_std, size=X.shape)


def run_single_clustering_on_perturbation(
    X, y_true, noise_std, best_params, K=2, rng=None, random_state=1
):
    """Run one perturbation experiment using spectral clustering.

    Parameters
    ----------
    X : ndarray
        Original data matrix.
    y_true : array-like
        Ground-truth labels used for evaluation/alignment.
    noise_std : float
        Noise level passed to :func:`perturb_data`.
    best_params : dict
        Parameters describing the similarity graph to construct.
    K : int, default=2
        Number of clusters (eigenvectors) to request from spectral clustering.
    rng : np.random.RandomState, optional
        Random generator used to perturb the data.

    Returns
    -------
    dict
        Contains the perturbed data, aligned labels, confusion matrix,
        accuracy, eigenvalues, and eigenvectors from the clustering run.
    """
    if rng is None:
        rng = np.random.RandomState(random_state)

    Xp = perturb_data(X, noise_std, rng=rng)
    distance_matrix = get_distance_matrix_from_data(Xp)

    similarity_matrix = get_similarity_matrix_from_distance_matrix(
        distance_matrix,
        sim_graph_type=best_params["sim_graph_type"],
        knn=best_params["knn"],
        sigma=best_params.get("sigma", 0),
        mutual_knn=best_params.get("mutual_knn", 0),
        epsilon=best_params.get("epsilon", 0),
    )

    result = Spectral_Clustering(
        similarity_matrix,
        K=K,
        random_state=random_state,
    )

    cm_after, labels_aligned = match_labels(y_true, result.clustering_model.labels_)

    return {
        "Xp": Xp,
        "labels_aligned": labels_aligned,
        "confusion_matrix": cm_after,
        "accuracy": float((y_true == labels_aligned).mean()),
        "eigenvalues": result.eigenvalues,
        "eigenvectors": result.eigenvectors,
    }


def run_experiment(
    X,
    y,
    best_params,
    n_runs=1,
    noise_std=0.5,
    K=2,
    seed=None,
    return_all=False,
    random_state=1,
):
    """Evaluate robustness by repeating noisy spectral clustering runs.

    Parameters
    ----------
    X : ndarray
        Original data used for clustering.
    y : array-like
        Ground-truth labels for evaluation.
    best_params : dict
        Graph-construction parameters passed to
        :func:`get_similarity_matrix_from_distance_matrix`.
    n_runs : int, default=1
        Number of perturbation trials to execute.
    noise_std : float, default=0.5
        Standard deviation of the Gaussian noise applied in each trial.
    K : int, default=2
        Number of clusters requested in spectral clustering.
    seed : int, optional
        Seed for the top-level RNG; defaults to ``random_state`` when ``None``.
    return_all : bool, default=False
        When ``True`` returns the list of all per-run dictionaries; otherwise
        returns aggregate statistics.
    random_state : int, default=1
        Fallback seed for RNG creation.

    Returns
    -------
    dict or list
        Per-run results when ``return_all`` is ``True``; otherwise a summary
        dictionary with accuracies and run metadata.
    """
    if seed is None:
        seed = random_state
    master_rng = np.random.RandomState(seed)

    results = []
    for i in range(n_runs):
        # create independent RNG per run for reproducibility
        run_rng = np.random.RandomState(master_rng.randint(0, 2**31 - 1))
        results.append(
            run_single_clustering_on_perturbation(
                X,
                y,
                noise_std,
                best_params,
                K=K,
                rng=run_rng,
                random_state=random_state,
            )
        )

    accuracy = np.array([r["accuracy"] for r in results])

    print(f"Runs: {n_runs}, noise_std: {noise_std}")
    print(f"Mean acc after  matching: {accuracy.mean():.4f} ± {accuracy.std():.4f}")

    if return_all:
        return results
    return {"results": results, "accuracy": accuracy}


def cross_validation_stability_test(
    distance_matrix,
    params,
    K_folds=10,
    random_state=19,
    info_score=normalized_mutual_info_score,
):
    """Evaluate spectral clustering stability via stratified K-fold splits.

    Parameters
    ----------
    similarity_matrix : ndarray
        Precomputed similarity/adjacency matrix for the full dataset (square).
    K_folds : int, default=5
        Number of stratified folds used in the stability test.
    K_clusters : int, default=4
        Number of clusters to request from :func:`Spectral_Clustering`.
    random_state : int, default=19
        Seed for the ``StratifiedKFold`` shuffling as well as per-fold clustering.
    info_score : callable, default=normalized_mutual_info_score
        Function applied to compare the per-fold labels with the reference
        partition. Larger values indicate closer agreement.

    Returns
    -------
    dict
        Dictionary containing the reference labels, per-fold scores, and their
        mean/std summary.
    """

    if (
        distance_matrix.ndim != 2
        or distance_matrix.shape[0] != distance_matrix.shape[1]
    ):
        raise ValueError("similarity_matrix must be a square matrix")
    if K_folds < 2:
        raise ValueError("K_folds must be at least 2")

    similarity_matrix = get_similarity_matrix_from_distance_matrix(
        distance_matrix,
        sim_graph_type=params["sim_graph_type"],
        knn=params["knn"],
        mutual_knn=params["mutual_knn"],
        sigma=params["sigma"],
        epsilon=params["epsilon"],
    )

    full_result = Spectral_Clustering(
        similarity_matrix,
        K=params["K"],
        random_state=random_state,
    )
    reference_labels = full_result.labels

    skf = StratifiedKFold(
        n_splits=K_folds,
        shuffle=True,
        random_state=random_state,
    )

    dummy_features = np.zeros((similarity_matrix.shape[0], 1))
    fold_scores = []

    # Replace train_idx with test_idx below to switch to small disjoint batches
    for fold_idx, (train_idx, test_idx) in enumerate(
        skf.split(dummy_features, reference_labels), start=1
    ):
        train_dist_matrix = distance_matrix[np.ix_(train_idx, train_idx)]

        train_matrix = get_similarity_matrix_from_distance_matrix(
            train_dist_matrix,
            sim_graph_type=params["sim_graph_type"],
            knn=params["knn"],
            mutual_knn=params["mutual_knn"],
            sigma=params["sigma"],
            epsilon=params["epsilon"],
        )

        fold_result = Spectral_Clustering(
            train_matrix,
            K=params["K"],
            random_state=random_state + fold_idx,
        )
        fold_labels = fold_result.labels

        print("Labels shape: ", fold_labels.shape)

        score = float(info_score(reference_labels[train_idx], fold_labels))
        fold_scores.append(score)
        print(f"Fold {fold_idx}: info_score(train vs full) = {score:.4f}")

    mean_score = float(np.mean(fold_scores)) if fold_scores else float("nan")
    std_score = float(np.std(fold_scores)) if fold_scores else float("nan")

    print(f"Mean info_score: {mean_score:.4f}")
    print(f"Std  info_score: {std_score:.4f}")

    return {
        "reference_labels": reference_labels,
        "fold_scores": fold_scores,
        "mean_score": mean_score,
        "std_score": std_score,
    }


def analyse_soft_spectral_clustering_stability(result):
    """Compute the mean posterior entropy for a clustering result.

    Parameters
    ----------
    result : SpectralClusteringResult
        Output from :func:`Spectral_Clustering` (ideally with ``soft=True`` to
        expose posterior probabilities).
    K : int
        Number of clusters (unused for now but kept for API compatibility).

    Returns
    -------
    float or None
        Average Shannon entropy over all samples when soft probabilities are
        available, otherwise ``None``.
    """

    probs = result.probs
    if probs is not None:
        probs_safe = np.clip(probs, 1e-10, 1.0)  # Avoid log(0)
        entropy_per_point = -np.sum(probs_safe * np.log2(probs_safe), axis=1)
        mean_entropy = np.mean(entropy_per_point)

    return mean_entropy


def drop_data_stability_test(
    distance_matrix,
    params,
    n_runs=20,
    alpha=0.2,
    random_state=19,
    info_score=normalized_mutual_info_score,
):
    """Evaluate stability by repeatedly dropping a fraction of samples.

    Each run removes ``alpha`` proportion of samples at random, fits spectral
    clustering on the remaining data, and compares the resulting labels to the
    reference partition obtained from the full dataset.

    Parameters
    ----------
    distance_matrix : ndarray
        Pairwise distance matrix for the full dataset (square).
    params : dict
        Graph-construction parameters (``sim_graph_type``, ``knn``, etc.) and the
        desired number of clusters ``K``.
    n_runs : int, default=20
        Number of random subsampling experiments to perform.
    alpha : float, default=0.2
        Fraction of samples to drop on each run. Must satisfy ``0 <= alpha < 1``.
    random_state : int, default=19
        Seed controlling the random subsampling and clustering randomness.
    info_score : callable, default=normalized_mutual_info_score
        Scoring function comparing the reference labels with run-specific labels.

    Returns
    -------
    dict
        Dictionary containing the reference labels, per-run scores, and their
        mean/std summary.
    """

    if (
        distance_matrix.ndim != 2
        or distance_matrix.shape[0] != distance_matrix.shape[1]
    ):
        raise ValueError("distance_matrix must be a square matrix")
    if not (0 <= alpha < 1):
        raise ValueError("alpha must be in the [0, 1) interval")
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")

    similarity_matrix = get_similarity_matrix_from_distance_matrix(
        distance_matrix,
        sim_graph_type=params["sim_graph_type"],
        knn=params["knn"],
        mutual_knn=params["mutual_knn"],
        sigma=params["sigma"],
        epsilon=params["epsilon"],
    )

    full_result = Spectral_Clustering(
        similarity_matrix,
        K=params["K"],
        random_state=random_state,
    )
    reference_labels = full_result.labels

    n_samples = distance_matrix.shape[0]
    rng = np.random.RandomState(random_state)
    keep_size = max(params["K"], int(round((1 - alpha) * n_samples)))
    keep_size = min(n_samples, keep_size)
    if keep_size <= 0:
        raise ValueError("alpha is too large; no samples would remain.")

    fold_scores = []

    for run_idx in range(1, n_runs + 1):
        keep_idx = np.sort(rng.choice(n_samples, size=keep_size, replace=True))

        train_dist_matrix = distance_matrix[np.ix_(keep_idx, keep_idx)]
        train_matrix = get_similarity_matrix_from_distance_matrix(
            train_dist_matrix,
            sim_graph_type=params["sim_graph_type"],
            knn=params["knn"],
            mutual_knn=params["mutual_knn"],
            sigma=params["sigma"],
            epsilon=params["epsilon"],
        )

        run_result = Spectral_Clustering(
            train_matrix,
            K=params["K"],
            random_state=random_state + run_idx,
        )
        run_labels = run_result.labels

        score = float(info_score(reference_labels[keep_idx], run_labels))
        fold_scores.append(score)
        print(
            f"Run {run_idx}: kept {keep_size}/{n_samples} samples, info_score = {score:.4f}"
        )

    mean_score = float(np.mean(fold_scores)) if fold_scores else float("nan")
    std_score = float(np.std(fold_scores)) if fold_scores else float("nan")

    print(f"Mean info_score: {mean_score:.4f}")
    print(f"Std  info_score: {std_score:.4f}")

    return {
        "reference_labels": reference_labels,
        "fold_scores": fold_scores,
        "mean_score": mean_score,
        "std_score": std_score,
    }


def plot_cross_validation_stability_test_result(
    stability_res_ami,
    stability_res_ars,
    metric_names=("AMI", "ARS"),
    palette=None,
    ax=None,
):
    """Visualize two cross-validation stability test outputs side by side."""

    if len(metric_names) != 2:
        raise ValueError("metric_names must provide exactly two labels")

    fold_scores_a = np.asarray(stability_res_ami["fold_scores"], dtype=float)
    fold_scores_b = np.asarray(stability_res_ars["fold_scores"], dtype=float)

    comparison_df = pd.DataFrame(
        {
            "Fold": np.tile(np.arange(1, fold_scores_a.size + 1), 2),
            "Score": np.concatenate([fold_scores_a, fold_scores_b]),
            "Metric": [metric_names[0]] * fold_scores_a.size
            + [metric_names[1]] * fold_scores_b.size,
        }
    )
    n_folds = int(comparison_df["Fold"].max()) if not comparison_df.empty else 0

    if palette is None:
        palette = {
            metric_names[0]: PRESENTATION_COLORS[0],
            metric_names[1]: PRESENTATION_COLORS[3],
        }

    sns.set_theme(style="whitegrid", context="notebook")
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
    else:
        fig = ax.figure

    sns.barplot(
        data=comparison_df,
        x="Fold",
        y="Score",
        hue="Metric",
        palette=palette,
        edgecolor="black",
        linewidth=0.3,
        ax=ax,
        dodge=True,
        alpha=0.9,
        width=0.7,
    )

    # Reduce wasted horizontal padding so bars use the available width.
    ax.margins(x=0.01)
    if n_folds > 0:
        ax.set_xlim(-0.5, n_folds - 0.5)

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Stability Score")
    ax.set_title("Stability Across Folds")
    ax.legend(bbox_to_anchor=(1.02, 0.5), frameon=False)

    mean_a = fold_scores_a.mean()
    mean_b = fold_scores_b.mean()

    ax.axhline(
        mean_a,
        color=palette[metric_names[0]],
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
    )
    ax.axhline(
        mean_b,
        color=palette[metric_names[1]],
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
    )

    # Place mean labels just outside the plotting area so they never overlap the
    # last bars and remain separated even when means are close.
    text_x = 1.02
    ax.text(
        text_x,
        0.98,
        f"{metric_names[0]} mean = {mean_a:.2f}",
        transform=ax.transAxes,
        color=palette[metric_names[0]],
        va="top",
        ha="left",
        fontsize=9,
        fontweight="semibold",
        clip_on=False,
    )
    ax.text(
        text_x,
        0.92,
        f"{metric_names[1]} mean = {mean_b:.2f}",
        transform=ax.transAxes,
        color=palette[metric_names[1]],
        va="top",
        ha="left",
        fontsize=9,
        fontweight="semibold",
        clip_on=False,
    )

    sns.despine(trim=True)
    if ax is None:
        fig.tight_layout(rect=(0, 0, 0.82, 1))
    else:
        fig.tight_layout()
    return ax
