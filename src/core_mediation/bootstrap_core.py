# src/core_mediation/bootstrap_core.py
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.core_mediation.constants import (
    COL_BOOT_SE,
    COL_KEPUTUSAN,
    COL_LLCI,
    COL_POINT_EST,
    COL_PREDIKTOR,
    COL_STATUS,
    COL_ULCI,
    LABEL_MEDIASI_TIDAK_TERBUKTI,
    LABEL_MEDIASI_TERBUKTI,
    LABEL_SIGNIFIKAN,
    LABEL_TIDAK_SIGNIFIKAN,
)


def _fit_ols_minimal(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit OLS via ordinary least squares and return coefficient vector.

    Args:
        X: Design matrix (shape: N x K).
        y: Target vector (shape: N,).

    Returns:
        np.ndarray: Coefficient vector (shape: K,).
    """
    return np.linalg.lstsq(X, y, rcond=None)[0]


def bootstrap_iteration(
    df: pd.DataFrame,
    predictor_cols: List[str],
    mediator_col: str,
    dependen_col: str,
    seed: int,
) -> np.ndarray:
    """Execute a single bootstrap iteration for mediation analysis.

    Fits Sub-Structure 1 (X -> M) and Sub-Structure 2 (X, M -> Y) on a
    resampled dataset, then computes the indirect effect (a * b) for each
    predictor.

    Args:
        df: Source dataframe.
        predictor_cols: Independent variable column names.
        mediator_col: Mediator column name.
        dependen_col: Dependent variable column name.
        seed: Integer seed for deterministic resampling.

    Returns:
        np.ndarray: Indirect effects (shape: n_predictors,).
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    n: int = len(df)
    idx: np.ndarray = rng.integers(0, n, size=n)

    df_boot: pd.DataFrame = df.iloc[idx].reset_index(drop=True)

    # Sub-Structure 1: X -> M
    X1: np.ndarray = sm.add_constant(
        df_boot[predictor_cols].values, has_constant="add"
    )
    y_m: np.ndarray = df_boot[mediator_col].values
    coef_a: np.ndarray = _fit_ols_minimal(X1, y_m)  # shape: (p + 1,)

    # Sub-Structure 2: X, M -> Y
    full_cols: List[str] = predictor_cols + [mediator_col]
    X2: np.ndarray = sm.add_constant(
        df_boot[full_cols].values, has_constant="add"
    )
    y_y: np.ndarray = df_boot[dependen_col].values
    coef_b: np.ndarray = _fit_ols_minimal(X2, y_y)  # shape: (p + 2,)

    # Mediator coefficient is the last element (mediator appended last,
    # constant prepended by sm.add_constant).
    b_mediator: float = float(coef_b[-1])

    # Predictor coefficients exclude the intercept (first element).
    a_predictors: np.ndarray = coef_a[1:]

    ie: np.ndarray = a_predictors * b_mediator
    return ie.astype(np.float64)


def run_bootstrap(
    df: pd.DataFrame,
    predictor_cols: List[str],
    mediator_col: str,
    dependen_col: str,
    n_iterations: int,
    seed: int,
    n_jobs: int = -1,
) -> np.ndarray:
    """Execute bootstrap iterations with deterministic seeding per iteration.

    Each iteration receives a unique derived seed to guarantee full
    reproducibility across parallel and sequential execution.

    Args:
        df: Source dataframe.
        predictor_cols: Independent variable column names.
        mediator_col: Mediator column name.
        dependen_col: Dependent variable column name.
        n_iterations: Number of bootstrap iterations.
        seed: Master random seed.
        n_jobs: Number of parallel jobs (-1 for all CPUs, 1 for sequential).

    Returns:
        np.ndarray: Bootstrap results (shape: n_iterations x n_predictors).
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    iteration_seeds: np.ndarray = rng.integers(
        0, 2**32, size=n_iterations, dtype=np.uint32
    )

    _run_iter = partial(
        bootstrap_iteration,
        df,
        predictor_cols,
        mediator_col,
        dependen_col,
    )

    results: List[np.ndarray]
    if n_jobs == 1:
        results = [_run_iter(int(s)) for s in iteration_seeds]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_run_iter)(int(s)) for s in iteration_seeds
        )

    return np.vstack(results)


def compute_ci_statistics(
    bootstrap_array: np.ndarray,
    predictor_names: List[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute point estimate, bootstrapped SE, and CI for each predictor.

    Args:
        bootstrap_array: Bootstrap results (shape: n_iterations x n_predictors).
        predictor_names: Names corresponding to each column.
        alpha: Significance level (e.g. 0.05 for 95% CI).

    Returns:
        pd.DataFrame: CI table (shape: n_predictors x 6).
    """
    lower_pct: float = alpha / 2 * 100
    upper_pct: float = (1 - alpha / 2) * 100

    rows: List[dict[str, Any]] = []
    for i, name in enumerate(predictor_names):
        col: np.ndarray = bootstrap_array[:, i]
        point_est: float = float(np.mean(col))
        boot_se: float = float(np.std(col, ddof=1))
        llci: float = float(np.percentile(col, lower_pct))
        ulci: float = float(np.percentile(col, upper_pct))

        is_significant: bool = not (llci <= 0.0 <= ulci)
        status: str = LABEL_SIGNIFIKAN if is_significant else LABEL_TIDAK_SIGNIFIKAN
        keputusan: str = (
            LABEL_MEDIASI_TERBUKTI if is_significant else LABEL_MEDIASI_TIDAK_TERBUKTI
        )

        rows.append(
            {
                COL_PREDIKTOR: name,
                COL_POINT_EST: round(point_est, 4),
                COL_BOOT_SE: round(boot_se, 4),
                COL_LLCI: round(llci, 4),
                COL_ULCI: round(ulci, 4),
                COL_STATUS: status,
                COL_KEPUTUSAN: keputusan,
            }
        )

    return pd.DataFrame(rows)


def export_mediation_excel(df: pd.DataFrame, output_path: Path) -> None:
    """Export mediation results to Excel.

    Args:
        df: Mediation results dataframe.
        output_path: Target Excel file path.
    """
    df.to_excel(output_path, index=False)


def export_bootstrap_array(bootstrap_array: np.ndarray, output_path: Path) -> None:
    """Export raw bootstrap array to .npy file.

    Args:
        bootstrap_array: Bootstrap results array.
        output_path: Target .npy file path.
    """
    np.save(output_path, bootstrap_array)
