import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

from spectral_clustering import *


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

    res, evects, evalues = Spectral_Clustering(
        similarity_matrix, K=K, random_state=random_state
    )

    cm_before = confusion_matrix(y_true, res.labels_)
    cm_after, labels_aligned = match_labels(y_true, res.labels_)

    return {
        "Xp": Xp,
        "labels_aligned": labels_aligned,
        "confusion_matrix": cm_after,
        "accuracy": float((y_true == labels_aligned).mean()),
        "eigenvalues": evalues,
        "eigenvectors": evects,
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
