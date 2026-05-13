import argparse
import sys
from pathlib import Path

# 1. Wajib jalankan sys.path ini PALING PERTAMA sebelum import dari 'src'
sys.path.append(str(Path(__file__).resolve().parent))

# 2. Gunakan HANYA SATU PINTU import (karena src/__init__.py sudah rapi)
from src import (
    initialize_workspace, 
    estimate_antecedent_effects, 
    estimate_mediator_outcomes, 
    validate_hypotheses,
    compute_mediation_bootstrap,
    compute_decomposition,
    compute_path_diagram,
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
        choices=["initialize_workspace", "estimate_antecedent_effects", "estimate_mediator_outcomes", "validate_hypotheses", "compute_mediation_bootstrap", "compute_decomposition", "compute_path_diagram", "all"],
        default="all",
        help=(
            "Excetion module available:\n"
            "    initialize_workspace: Setup Arsitektur Repordiksibilitas (Config dan Folder)\n"
            "    estimate_antecedent_effects: Estimasi Sub-Struktur 1 (X -> M) + Robust SE \n"
            "    estimate_mediator_outcomes: Estimasi Sub-Struktur 2 (X, M -> Y)"
            "    validate_hypotheses: Evaluasi Hipotesis"
            "    compute_mediation_bootstrap: Bootstraping"
            "    compute_decomposition: Kalkulasi Dekomposisi Efek"
            "    compute_path_diagram: !!!"
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

    if args.fase in ["validate_hypotheses", "all"]:
        print("\n[START] EVALUASI HIPOTESIS")
        try:
            validate_hypotheses()
            print("[SUCCESS]")
        except Exception as e:
            print(f"[ERROR]: {e}")
            sys.exit(1)

    if args.fase in ["compute_mediation_bootstrap", "all"]:
        print("\n[START] BOOTSTRAPING")
        try:
            compute_mediation_bootstrap()
            print("[SUCCESS]")
        except Exception as e:
            print(f"[ERROR]:{e}")
            sys.exit(1)

    if args.fase in ["compute_decomposition", "all"]:
        print("\n[START] DECOMPOSITION")
        try:
            compute_decomposition()
            print("[SUCCESS]")
        except Exception as e:
            print(f"[ERROR]:{e}")
            sys.exit(1)

    if args.fase in ["compute_path_diagram", "all"]:
        print("\n[START] DIAGRAM PATH")
        try:
            compute_path_diagram()
            print("[SUCCESS]")
        except Exception as e:
            print(f"[ERROR]:{e}")
            sys.exit(1)

    print("\n" + "="*60)
    print("[DONE]")
    print("="*60)

if __name__ == "__main__":
    main()
    
