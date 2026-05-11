# src/core_reproducibility/config_builder.py
from __future__ import annotations

from typing import Any, Dict

from src.core_reproducibility.constants import (
    ANALISIS_KEY,
    ARSITEKTUR_MODEL_KEY,
    JUMLAH_BOOTSTRAP_KEY,
    KUNCI_RANDOM_SEED_KEY,
    OUTPUT_FIGURES_KEY,
    OUTPUT_TABLES_KEY,
    PATHS_KEY,
    REPORTS_FIGURES_KEY,
    REPORTS_TABLES_KEY,
    REPRODUCIBILITY_KEY,
    ROOT_PATHS_KEY,
    SIGNIFIKANSI_ALPHA_KEY,
)
from src.core_reproducibility.schema import AnalisisConfig


class ConfigBuildError(Exception):
    """Raised when the application configuration cannot be resolved to a valid
    analisis configuration.  Eliminates raw :class:`KeyError` surfacing to
    callers.
    """
    pass


def _resolve_analisis_paths(app_config: Dict[str, Any]) -> Dict[str, str]:
    """Resolve the analisis output paths using the nested SSOT.

    The nested ``reproducibility.analisis.paths`` is the single source of
    truth.  If it is absent, the function falls back to the root ``paths``
    section using a well-defined key mapping.  Any resolution failure raises
    :class:`ConfigBuildError` with a descriptive message instead of a raw
    :class:`KeyError`.

    Args:
        app_config: Raw application configuration dictionary.

    Returns:
        Dictionary containing ``output_tables`` and ``output_figures`` paths.
    """
    repro_section: Dict[str, Any] | None = app_config.get(REPRODUCIBILITY_KEY)
    if repro_section is None:
        raise ConfigBuildError(
            f"Missing required top-level key '{REPRODUCIBILITY_KEY}' in configuration."
        )

    analisis_section: Dict[str, Any] | None = repro_section.get(ANALISIS_KEY)
    if analisis_section is None:
        raise ConfigBuildError(
            f"Missing required key '{ANALISIS_KEY}' under '{REPRODUCIBILITY_KEY}'."
        )

    # Single Source of Truth: nested paths
    nested_paths: Dict[str, Any] | None = analisis_section.get(PATHS_KEY)
    if nested_paths is not None:
        try:
            return {
                OUTPUT_TABLES_KEY: nested_paths[OUTPUT_TABLES_KEY],
                OUTPUT_FIGURES_KEY: nested_paths[OUTPUT_FIGURES_KEY],
            }
        except KeyError as exc:
            raise ConfigBuildError(
                f"Missing expected path key in nested '{PATHS_KEY}': {exc}"
            ) from exc

    # Fallback: root paths (backward-compatibility layer)
    root_paths: Dict[str, Any] | None = app_config.get(ROOT_PATHS_KEY)
    if root_paths is None:
        raise ConfigBuildError(
            f"Nested '{PATHS_KEY}' not found under '{REPRODUCIBILITY_KEY}.{ANALISIS_KEY}' "
            f"and fallback '{ROOT_PATHS_KEY}' is also missing."
        )

    try:
        return {
            OUTPUT_TABLES_KEY: root_paths[REPORTS_TABLES_KEY],
            OUTPUT_FIGURES_KEY: root_paths[REPORTS_FIGURES_KEY],
        }
    except KeyError as exc:
        raise ConfigBuildError(
            f"Missing expected fallback path key in root '{ROOT_PATHS_KEY}': {exc}"
        ) from exc


def build_analisis_config(app_config: Dict[str, Any]) -> AnalisisConfig:
    """Build the canonical analisis configuration dictionary.

    Reads from ``app_config["reproducibility"]["analisis"]`` and injects the
    resolved output paths.  All string keys are sourced from
    :mod:`src.core_reproducibility.constants` to guarantee zero hard-coding
    in business logic.

    Args:
        app_config: Raw application configuration dictionary.

    Returns:
        Canonical :class:`AnalisisConfig` dictionary.
    """
    repro_section: Dict[str, Any] | None = app_config.get(REPRODUCIBILITY_KEY)
    if repro_section is None:
        raise ConfigBuildError(
            f"Missing required top-level key '{REPRODUCIBILITY_KEY}' in configuration."
        )

    analisis_section: Dict[str, Any] | None = repro_section.get(ANALISIS_KEY)
    if analisis_section is None:
        raise ConfigBuildError(
            f"Missing required key '{ANALISIS_KEY}' under '{REPRODUCIBILITY_KEY}'."
        )

    paths: Dict[str, str] = _resolve_analisis_paths(app_config)

    try:
        return AnalisisConfig(
            signifikansi_alpha=analisis_section[SIGNIFIKANSI_ALPHA_KEY],
            jumlah_bootstrap=analisis_section[JUMLAH_BOOTSTRAP_KEY],
            kunci_random_seed=analisis_section[KUNCI_RANDOM_SEED_KEY],
            arsitektur_model=analisis_section[ARSITEKTUR_MODEL_KEY],
            paths=paths,
        )
    except KeyError as exc:
        raise ConfigBuildError(
            f"Missing required analisis parameter key: {exc}"
        ) from exc
