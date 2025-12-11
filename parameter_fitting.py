import numpy as np
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
)
from sklearn.model_selection import ParameterGrid

from spectral_clustering import *


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
    dict or None
        Parameter dictionary that produced the largest eigengap, or ``None`` if
        no configuration succeeded.
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
            result = Spectral_Clustering(
                similarity_matrix,
                K=params["K"],
                random_state=random_state,
            )
        except ValueError as exc:
            if "connected components" in str(exc).lower():
                continue
            raise

        eigengap = get_eigengap(result.eigenvalues)

        # print(params)
        # print("Eigengap: ", eigengap)

        if eigengap > best_gap:
            best_gap = eigengap
            best_params = params.copy()

            # Attempt to refine K based on the detected eigengap.
            try:
                estimated_k = estimate_k_from_eigengap(result.eigenvalues)
                Spectral_Clustering(
                    similarity_matrix,
                    K=estimated_k,
                    random_state=random_state,
                )
                best_params["K"] = estimated_k
                # print("Average eigenvalue: ", np.mean(result.eigenvalues[:estimated_k]))
                # print("Estimated K: ", estimated_k)

            except ValueError as exc:
                if "connected components" in str(exc).lower():
                    continue
                raise

        # print()

    return best_params


# NMI
def find_best_params_with_seed_iteration_nmi(
    distance_matrix,
    param_grid,
    n_runs=5,
    random_state=1,
    info_score=normalized_mutual_info_score,
):
    """Return the parameter set with the most stable labels across seeds.

    Parameters
    ----------
    distance_matrix : ndarray
        Pairwise distances between samples used to build similarity graphs.
    param_grid : dict or list of dicts
        Specification of parameter combinations compatible with ``ParameterGrid``.
    n_runs : int, default=5
        Number of random seeds to evaluate per configuration.
    random_state : int, default=1
        Base seed; individual runs offset from this value to ensure reproducibility.
    info_score : callable, default=normalized_mutual_info_score
        Function applied to two label arrays to quantify their agreement. Must
        return a scalar where larger values indicate greater similarity.

    Returns
    -------
    dict or None
        Parameter dictionary that yielded the highest mean NMI across seeds, or
        ``None`` if no configuration succeeded.
    """

    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")

    best_stability = -np.inf
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

        labels_per_seed = []
        invalid_config = False

        for run_idx in range(n_runs):
            seed = random_state + run_idx
            try:
                result = Spectral_Clustering(
                    similarity_matrix,
                    K=params["K"],
                    random_state=seed,
                )
            except ValueError as exc:
                if "connected components" in str(exc).lower():
                    invalid_config = True
                    break
                raise

            labels_per_seed.append(result.labels)

        if invalid_config or not labels_per_seed:
            continue

        base_labels = labels_per_seed[0]
        stability_scores = [
            info_score(base_labels, labels) for labels in labels_per_seed[1:]
        ]

        if stability_scores:
            stability_scores = np.asarray(stability_scores)
            mean_nmi = float(stability_scores.mean())
            std_nmi = float(stability_scores.std())
        else:
            mean_nmi = 1.0
            std_nmi = 0.0

        print(params)
        print(f"Mean NMI: {mean_nmi:.4f} ± {std_nmi:.4f}")
        print()

        if mean_nmi > best_stability:
            best_stability = mean_nmi
            best_params = params.copy()

    return best_params


# AMI
def find_best_params_with_seed_iteration_ami(
    distance_matrix,
    param_grid,
    n_runs=5,
    random_state=1,
    info_score=adjusted_mutual_info_score,
):
    """Return the parameter set with the most stable labels across seeds (AMI).

    Parameters
    ----------
    distance_matrix : ndarray
        Pairwise distances between samples used to build similarity graphs.
    param_grid : dict or list of dicts
        Specification of parameter combinations compatible with ``ParameterGrid``.
    n_runs : int, default=5
        Number of random seeds to evaluate per configuration.
    random_state : int, default=1
        Base seed; individual runs offset from this value to ensure reproducibility.
    info_score : callable, default=adjusted_mutual_info_score
        Function applied to two label arrays to quantify their agreement. Must
        return a scalar where larger values indicate greater similarity.

    Returns
    -------
    dict or None
        Parameter dictionary that yielded the highest mean AMI across seeds, or
        ``None`` if no configuration succeeded.
    """

    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")

    best_stability = -np.inf
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

        labels_per_seed = []
        invalid_config = False

        for run_idx in range(n_runs):
            seed = random_state + run_idx
            try:
                result = Spectral_Clustering(
                    similarity_matrix,
                    K=params["K"],
                    random_state=seed,
                )
            except ValueError as exc:
                if "connected components" in str(exc).lower():
                    invalid_config = True
                    break
                raise

            labels_per_seed.append(result.labels)

        if invalid_config or not labels_per_seed:
            continue

        base_labels = labels_per_seed[0]
        stability_scores = [
            info_score(base_labels, labels) for labels in labels_per_seed[1:]
        ]

        if stability_scores:
            stability_scores = np.asarray(stability_scores)
            mean_ami = float(stability_scores.mean())
            std_ami = float(stability_scores.std())
        else:
            mean_ami = 1.0
            std_ami = 0.0

        print(params)
        print(f"Mean AMI: {mean_ami:.4f} ± {std_ami:.4f}")
        print()

        if mean_ami > best_stability:
            best_stability = mean_ami
            best_params = params.copy()

    return best_params


# ARS
def find_best_params_with_seed_iteration_ars(
    distance_matrix,
    param_grid,
    n_runs=5,
    random_state=1,
    info_score=adjusted_rand_score,
):
    """Return the parameter set with the most stable labels across seeds (ARS).

    Parameters
    ----------
    distance_matrix : ndarray
        Pairwise distances between samples used to build similarity graphs.
    param_grid : dict or list of dicts
        Specification of parameter combinations compatible with ``ParameterGrid``.
    n_runs : int, default=5
        Number of random seeds to evaluate per configuration.
    random_state : int, default=1
        Base seed; individual runs offset from this value to ensure reproducibility.
    info_score : callable, default=adjusted_rand_score
        Function applied to two label arrays to quantify their agreement. Must
        return a scalar where larger values indicate greater similarity.

    Returns
    -------
    dict or None
        Parameter dictionary that yielded the highest mean ARS across seeds, or
        ``None`` if no configuration succeeded.
    """

    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")

    best_stability = -np.inf
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

        labels_per_seed = []
        invalid_config = False

        for run_idx in range(n_runs):
            seed = random_state + run_idx
            try:
                result = Spectral_Clustering(
                    similarity_matrix,
                    K=params["K"],
                    random_state=seed,
                )
            except ValueError as exc:
                if "connected components" in str(exc).lower():
                    invalid_config = True
                    break
                raise

            labels_per_seed.append(result.labels)

        if invalid_config or not labels_per_seed:
            continue

        base_labels = labels_per_seed[0]
        stability_scores = [
            info_score(base_labels, labels) for labels in labels_per_seed[1:]
        ]

        if stability_scores:
            stability_scores = np.asarray(stability_scores)
            mean_ars = float(stability_scores.mean())
            std_ars = float(stability_scores.std())
        else:
            mean_ars = 1.0
            std_ars = 0.0

        print(params)
        print(f"Mean ARS: {mean_ars:.4f} ± {std_ars:.4f}")
        print()

        if mean_ars > best_stability:
            best_stability = mean_ars
            best_params = params.copy()

    return best_params
