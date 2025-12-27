import numpy as np
import pandas as pd

# constructors for simple graphs


def W_complete_graph(n: int) -> np.ndarray:
    """Complete graph with unit weights, no self-loops."""
    W = np.ones((n, n))
    np.fill_diagonal(W, 0.0)
    return W


def W_path_graph(n: int) -> np.ndarray:
    """Path graph: 1-2-3-...-n, unit weights."""
    W = np.zeros((n, n))
    for i in range(n - 1):
        W[i, i + 1] = 1.0
        W[i + 1, i] = 1.0
    return W


def build_laplacians_from_W(W: np.ndarray):
    """If you already have this in your code, just use that instead."""
    d = W.sum(axis=1)
    D = np.diag(d)
    L = D - W
    # Random-walk Laplacian
    D_inv = np.diag(1.0 / d)
    L_rw = np.eye(W.shape[0]) - D_inv @ W
    return D, L, L_rw


def center_signal(f: np.ndarray) -> np.ndarray:
    """Center a signal on the nodes: subtract mean so sum_i f_i = 0."""
    return f - f.mean()


def empirical_variance(f: np.ndarray) -> float:
    """Var(f) = (1/n) * sum_i f_i^2, assuming f is already centered."""
    n = f.shape[0]
    return float((f @ f) / n)


def laplacian_energy(f: np.ndarray, L: np.ndarray) -> float:
    """Energy(f) = f^T L f."""
    return float(f.T @ (L @ f))


def experiment_variance_vs_energy(n=10, n_trials=5):
    results = []

    graph_builders = {
        "complete": W_complete_graph,
        "path": W_path_graph,
    }

    for name, builder in graph_builders.items():
        W = builder(n)
        D, L, L_rw = build_laplacians_from_W(
            W
        )  # I think we do not have any other functions to build laplkacians

        for trial in range(n_trials):
            # Random Gaussian signal
            f = np.random.randn(n)
            f_c = center_signal(f)
            var_f = empirical_variance(f_c)
            energy_f = laplacian_energy(f_c, L)
            ratio = energy_f / var_f if var_f > 0 else np.nan

            results.append(
                {
                    "graph": name,
                    "trial": trial,
                    "var": var_f,
                    "energy": energy_f,
                    "ratio": ratio,
                }
            )

    df = pd.DataFrame(results)
    summary = df.groupby("graph")[["var", "energy", "ratio"]].mean()
    print(summary)
    return df, summary
