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
    run_nlp_frequential,
    run_nlp_sentiment,
    run_nlp_aggregation,
    run_nlp_llm_synthesis,
)


def main() -> None:
    """Orkestrator Utama untuk Menjalankan Seluruh Pipeline Analisis Tesis."""
    parser = argparse.ArgumentParser(
        description="Main Orchestrator - Thesis Data Synthesis & NLP Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--fase",
        type=str,
        choices=[
            "initialize_workspace",
            "run_nlp_preprocessing",
            "run_nlp_frequential",
            "run_nlp_sentiment",
            "run_nlp_aggregation",
            "run_nlp_llm_synthesis",
            "estimate_antecedent_effects",
            "estimate_mediator_outcomes",
            "validate_hypotheses",
            "compute_mediation_bootstrap",
            "compute_decomposition",
            "run_path_diagram",
            "run_radar_chart",
            "all",
        ],
        default="all",
        help=(
            "Execution modules available:\n"
            "    initialize_workspace: Setup Arsitektur Reproduksibilitas\n"
            "    run_nlp_preprocessing: NLP Text Preprocessing (4 tahap)\n"
            "    run_nlp_frequential  : NLP Frequential Analysis (N-Gram & WordCloud)\n"
            "    run_nlp_sentiment    : NLP Sentiment Analysis (IndoBERT Client)\n"
            "    run_nlp_aggregation  : Master Aggregation & Sentiment Scoring\n"
            "    run_nlp_llm_synthesis: LLM Academic Narrative Synthesis (OpenRouter)\n"
            "    estimate_antecedent_effects : Estimasi Regresi (X -> M)\n"
            "    estimate_mediator_outcomes  : Estimasi Regresi (X, M -> Y)\n"
            "    validate_hypotheses  : Evaluasi & Validasi Hipotesis SEM\n"
            "    compute_mediation_bootstrap : Bootstrapping Efek Mediasi\n"
            "    compute_decomposition: Dekomposisi Efek (Direct, Indirect, Total)\n"
            "    run_path_diagram     : Generasi Visualisasi Diagram Path\n"
            "    run_radar_chart      : Generasi Visualisasi Radar Chart Dimensi\n"
            "    all                  : Eksekusi Seluruh Pipeline Secara Utuh"
        )
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ANALISIS PIPELINE INITIALIZED")
    print("=" * 60)

    phases = [
        ("initialize_workspace", initialize_workspace, "SETUP ARSITEKTUR"),
        ("run_nlp_preprocessing", run_nlp_preprocessing, "NLP TEXT PREPROCESSING"),
        ("run_nlp_frequential", run_nlp_frequential, "NLP FREQUENTIAL ANALYSIS"),
        ("run_nlp_sentiment", run_nlp_sentiment, "NLP SENTIMENT ANALYSIS"),
        ("run_nlp_aggregation", run_nlp_aggregation, "MASTER AGGREGATION & SCORING"),
        ("run_nlp_llm_synthesis", run_nlp_llm_synthesis, "LLM ACADEMIC NARRATIVE SYNTHESIS"),
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
                print(f"[FAILED] Error pada fase {phase_label}: {str(e)}")
                sys.exit(1)

    print("\n" + "=" * 60)
    print("[DONE] SELURUH PIPELINE BERHASIL DIEKSEKUSI")
    print("=" * 60)


if __name__ == "__main__":
    main()
