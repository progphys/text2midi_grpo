from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from scripts.infer import parse_args

    args = parse_args()
    from text2midi.infer_app import run_inference

    run_inference(args)
