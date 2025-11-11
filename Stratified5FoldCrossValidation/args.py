import argparse
import pathlib


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV with columns: image_path, KL_grade")
    p.add_argument("--out", default="outputs", help="Output dir")
    p.add_argument("--test_size", type=float, default=0.20)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int,default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    args=p.parse_args()
    pathlib.Path(args.out).mkdir(parents=True,exist_ok=True)
    return args
