# src/core_mediation/bootstrap_core.py
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _fit_ols_minimal(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit OLS via least squares and return coefficients.

    Returns:
        np.ndarray: Coefficient vector (shape: K+1,).
    """
    return np.linalg.lstsq(X, y, rcond=None)[0]


def bootstrap_iteration(
    df: pd.DataFrame,
    predictor_cols: list[str],
    mediator_col: str,
    dependen_col: str,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Single bootstrap iteration: resample, fit Sub-1 & Sub-2, compute IE.

    Returns:
        tuple: (IE_people, IE_process, IE_physical_evidence).
    """
    n = len(df)
    idx = rng.integers(0, n, size=n)

    df_boot = df.iloc[idx].reset_index(drop=True)

    # Sub-1: X -> M (coefficients a for each predictor)
    X_sub1 = sm.add_constant(df_boot[predictor_cols].values, has_constant="add")
    y_m = df_boot[mediator_col].values
    coef_a = _fit_ols_minimal(X_sub1, y_m)

    # Sub-2: X, M -> Y (coefficient b for mediator)
    full_cols = predictor_cols + [mediator_col]
    X_sub2 = sm.add_constant(df_boot[full_cols].values, has_constant="add")
    y_y = df_boot[dependen_col].values
    coef_b = _fit_ols_minimal(X_sub2, y_y)

    # Indirect Effect: a_i * b_mediator
    # coef_a[0] is intercept; coef_a[1:] are predictor effects
    # coef_b[-1] is mediator effect (last coefficient after predictors)
    ie = coef_a[1:] * coef_b[-1]

    return (float(ie[0]), float(ie[1]), float(ie[2]))


def run_bootstrap_parallel(
    df: pd.DataFrame,
    predictor_cols: list[str],
    mediator_col: str,
    dependen_col: str,
    n_iterations: int,
    seed: int,
    n_jobs: int = -1,
) -> np.ndarray:
    """Execute bootstrap iterations in parallel.

    Returns:
        np.ndarray: Bootstrap results (shape: n_iterations x 3).
    """
    rng = np.random.default_rng(seed)

    from joblib import Parallel, delayed

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(bootstrap_iteration)(
            df, predictor_cols, mediator_col, dependen_col, rng
        )
        for _ in range(n_iterations)
    )

    return np.array(results, dtype=np.float64)


def compute_ci_statistics(
    bootstrap_array: np.ndarray, predictor_names: list[str]
) -> pd.DataFrame:
    """Compute point estimate, boot SE, and 95% CI for each predictor.

    Returns:
        pd.DataFrame: CI table (shape: 3 x 6).
    """
    rows: list[dict[str, Any]] = []

    for i, name in enumerate(predictor_names):
        col = bootstrap_array[:, i]
        point_est = float(np.mean(col))
        boot_se = float(np.std(col, ddof=1))
        llci, ulci = float(np.percentile(col, 2.5)), float(
            np.percentile(col, 97.5)
        )

        status = "Signifikan" if not (llci <= 0 <= ulci) else "Tidak Signifikan"
        keputusan = "Mediasi Terbukti" if status == "Signifikan" else "Mediasi Tidak Terbukti"

        rows.append(
            {
                "Prediktor": name,
                "Point_Estimate": round(point_est, 4),
                "Boot_SE": round(boot_se, 4),
                "LLCI_95": round(llci, 4),
                "ULCI_95": round(ulci, 4),
                "Status": status,
                "Keputusan": keputusan,
            }
        )

    return pd.DataFrame(rows)


def export_mediation_excel(df: pd.DataFrame, output_path: str) -> None:
    """Export mediation results to Excel.

    Args:
        df: Mediation results dataframe.
        output_path: Target Excel file path.
    """
    df.to_excel(output_path, index=False)


def export_bootstrap_array(bootstrap_array: np.ndarray, output_path: str) -> None:
    """Export raw bootstrap array to .npy file.

    Args:
        bootstrap_array: Bootstrap results array.
        output_path: Target .npy file path.
    """
    np.save(output_path, bootstrap_array)
