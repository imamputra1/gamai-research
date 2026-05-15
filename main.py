# main.py
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from src import (
    initialize_workspace,
    estimate_antecedent_effects,
    estimate_mediator_outcomes,
    validate_hypotheses,
    compute_mediation_bootstrap,
    compute_decomposition,
    run_path_diagram,
    run_radar_chart,
    run_nlp_preprocessing,
)


def main() -> None:
    """Orekestrator Utama untuk Run Analisis"""
    parser = argparse.ArgumentParser(
        description="Main Orekestrator",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--fase",
        type=str,
        choices=[
            "initialize_workspace",
            "estimate_antecedent_effects",
            "estimate_mediator_outcomes",
            "validate_hypotheses",
            "compute_mediation_bootstrap",
            "compute_decomposition",
            "run_path_diagram",
            "run_radar_chart",
            "run_nlp_preprocessing",
            "all",
        ],
        default="all",
        help=(
            "Execution modules available:\n"
            "    initialize_workspace: Setup Arsitektur Reproduksibilitas\n"
            "    estimate_antecedent_effects: Estimasi Sub-Struktur 1 (X -> M)\n"
            "    estimate_mediator_outcomes: Estimasi Sub-Struktur 2 (X, M -> Y)\n"
            "    validate_hypotheses: Evaluasi Hipotesis\n"
            "    compute_mediation_bootstrap: Bootstrapping\n"
            "    compute_decomposition: Kalkulasi Dekomposisi Efek\n"
            "    run_path_diagram: Injeksi Metrik Kausalitas\n"
            "    run_radar_chart: Konstruksi Radar Chart (TCR)\n"
            "    run_nlp_preprocessing: NLP Text Preprocessing (4 tahap)\n"
            "    all: Eksekusi pipeline utuh"
        )
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ANALISIS PIPELINE INITIALIZED")
    print("=" * 60)

    phases = [
        ("initialize_workspace", initialize_workspace, "SETUP ARSITEKTUR"),
        ("run_nlp_preprocessing", run_nlp_preprocessing, "NLP TEXT PREPROCESSING"),
        ("estimate_antecedent_effects", estimate_antecedent_effects, "ESTIMASI (X -> M)"),
        ("estimate_mediator_outcomes", estimate_mediator_outcomes, "ESTIMASI (X, M -> Y)"),
        ("validate_hypotheses", validate_hypotheses, "EVALUASI HIPOTESIS"),
        ("compute_mediation_bootstrap", compute_mediation_bootstrap, "BOOTSTRAPPING"),
        ("compute_decomposition", compute_decomposition, "DECOMPOSITION"),
        ("run_path_diagram", run_path_diagram, "DIAGRAM PATH"),
        ("run_radar_chart", run_radar_chart, "RADAR CHART"),
    ]

    for phase_key, phase_func, phase_label in phases:
        if args.fase in [phase_key, "all"]:
            print(f"\n[START] {phase_label}")
            try:
                phase_func()
                print("[SUCCESS]")
            except Exception as e:
                print(f"[ERROR]: {e}")
                sys.exit(1)

    print("\n" + "=" * 60)
    print("[DONE]")
    print("=" * 60)


if __name__ == "__main__":
    main()
