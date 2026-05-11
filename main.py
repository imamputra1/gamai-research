import argparse
import sys
from pathlib import Path

from src.core_mediation.orchestrator import run_fase_13_2

sys.path.append(str(Path(__file__).resolve().parent))

from src import initialize_workspace, estimate_antecedent_effects, estimate_mediator_outcomes, run_fase_13_1

def main() -> None:
    """Orekestrator Utama untuk Run Analisis"""
    parser = argparse.ArgumentParser(
        description="Main Orekestrator",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--fase",
        type=str,
        choices=["initialize_workspace", "estimate_antecedent_effects", "estimate_mediator_outcomes", "13.1", "13.2", "all"],
        default="all",
        help=(
            "Excetion module available:\n"
            "    initialize_workspace: Setup Arsitektur Repordiksibilitas (Config dan Folder)\n"
            "    estimate_antecedent_effects: Estimasi Sub-Struktur 1 (X -> M) + Robust SE \n"
            "    estimate_mediator_outcomes: Estimasi Sub-Struktur 2 (X, M -> Y)"
            "    13.1: Evaluasi Hipotesis"
            "    13.2: Bootstraping"
            "    all : Eksekusi pipeline Utuh"
        )
    )

    args = parser.parse_args()

    print("="*60)
    print("ANALISIS PIPELINE INITIALIZED")
    print("="*60)

    if args.fase in ["initialize_workspace", "all"]:
        print("\n[START] SETUP ARSITEKTUR")
        try:
            initialize_workspace()
            print("[SUCCESS]")
        except Exception as e:
            print(f"[ERROR]: {e}")
            sys.exit(1)

    if args.fase in ["estimate_antecedent_effects", "all"]:
        print("\n[START] ESTIMASI (X -> M) RUN")
        try:
            estimate_antecedent_effects()
            print("[SUCCESS]}")
        except Exception as e:
            print(f"[ERROR]: {e}")
            sys.exit(1)

    if args.fase in ["estimate_mediator_outcomes", "all"]:
        print("\n[START] ESTIMASI (X, M -> Y) RUN")
        try:
            estimate_mediator_outcomes()
            print("[SUCCESS]")
        except Exception as e:
            print(f"[ERROR]: {e}")
            sys.exit(1)

    if args.fase in ["13.1", "all"]:
        print("\n[START] EVALUASI HIPOTESIS")
        try:
            run_fase_13_1()
            print("[SUCCESS]")
        except Exception as e:
            print(f"[ERROR]: {e}")
            sys.exit(1)

    if args.fase in ["13.2", "all"]:
        print("\n[START] BOOTSTRAPING")
        try:
            run_fase_13_2()
            print("[SUCCESS]")
        except Exception as e:
            print(f"[ERROR]:{e}")
            sys.exit(1)

    print("\n" + "="*60)
    print("[DONE]")
    print("="*60)

if __name__ == "__main__":
    main()
    
