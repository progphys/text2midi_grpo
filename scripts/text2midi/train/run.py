from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


if __name__ == "__main__":
    from scripts.train import parse_args

    args = parse_args()
    from text2midi.train_app import run_training

    run_training(args)
