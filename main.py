import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from src import run_fase_12_0, run_fase_12_1

def main() -> None:
    """Orekestrator Utama untuk Run Analisis"""
    parser = argparse.ArgumentParser(
        description="Main Orekestrator",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--fase",
        type=str,
        choices=["12.0", "12.1", "all"],
        default="all",
        help=(
            "Excetion module available:\n"
            "    12.0: Setup Arsitektur Repordiksibilitas (Config dan Folder)\n"
            "    12.1: Estimasi Sub-Struktur 1 (X -> M) + Robust SE \n"
            "    all : Eksekusi pipeline Utuh"
        )
    )

    args = parser.parse_args()

    print("="*60)
    print("ANALISIS PIPELINE INITIALIZED")
    print("="*60)

    if args.fase in ["12.0", "all"]:
        print("\n[START] SETUP ARSITEKTUR")
        try:
            run_fase_12_0()
            print("[SUCCESS]")
        except Exception as e:
            print(f"[ERROR]: {e}")
            sys.exit(1)

    if args.fase in ["12.1", "all"]:
        print("\n[START] ESTIMASI RUN")
        try:
            run_fase_12_1()
            print("[SUCCESS]}")
        except Exception as e:
            print(f"[ERROR]: {e}")
            sys.exit(1)

    print("\n" + "="*60)
    print("[DONE]")
    print("="*60)

if __name__ == "__main__":
    main()
    
