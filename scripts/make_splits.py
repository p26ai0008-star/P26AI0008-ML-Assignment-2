#!/usr/bin/env python3
"""Regenerate the Olivetti faces train/validation/test split.

This is the standalone version of the dataset-splitting cell at the top of
ML_assignment_P26AI0008.ipynb. The split is fully deterministic: it depends
only on the seed 260008 and on the identity labels of the scikit-learn
Olivetti faces dataset, so it produces exactly the indices stored in
data/olivetti_split.npz.

Per identity (10 images each, 40 identities):
    first 6 shuffled images  -> train   (240 images)
    next 2 shuffled images   -> val     (80 images)
    remaining 2 images       -> test    (80 images)

Usage:
    python scripts/make_splits.py [--out PATH]

Default output: data/olivetti_split.npz inside the repository root.
The Olivetti dataset (~6 MB) is downloaded automatically by scikit-learn on
first use and cached in ~/scikit_learn_data.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_olivetti_faces

SEED = 260008
REPO_ROOT = Path(__file__).resolve().parent.parent


def make_split():
    """Return (train_idx, val_idx, test_idx) exactly as in the notebook."""
    faces = fetch_olivetti_faces()
    y = faces.target

    rng = np.random.default_rng(SEED)

    train_idx, val_idx, test_idx = [], [], []

    for identity in np.unique(y):
        idx = np.where(y == identity)[0]
        rng.shuffle(idx)

        train_idx.extend(idx[:6])
        val_idx.extend(idx[6:8])
        test_idx.extend(idx[8:])

    return (np.array(train_idx), np.array(val_idx), np.array(test_idx))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "data" / "olivetti_split.npz"),
        help="output path of the .npz split file "
             "(default: data/olivetti_split.npz)",
    )
    args = parser.parse_args()

    train_idx, val_idx, test_idx = make_split()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    print(f"Saved split to {out_path}")
    print(f"Train: {len(train_idx)} | Validation: {len(val_idx)} | "
          f"Test: {len(test_idx)}")
    print("train_idx[:10]:", train_idx[:10].tolist())
    print("val_idx[:10]:", val_idx[:10].tolist())
    print("test_idx[:10]:", test_idx[:10].tolist())


if __name__ == "__main__":
    main()
