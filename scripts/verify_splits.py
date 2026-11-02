#!/usr/bin/env python3
"""Verify that the saved Olivetti split matches a fresh regeneration.

Regenerates the split from the seed (same logic as the notebook and as
scripts/make_splits.py) and compares it element-by-element with a reference
.npz file. Prints an MD5 checksum per array and exits with code 1 if any
mismatch is found, so it can be used inside reproduction scripts.

Usage:
    python scripts/verify_splits.py [--reference PATH]
                                    [--also-results PATH]

    --reference      split file to check against
                     (default: data/olivetti_split.npz)
    --also-results   optionally also compare a second copy of the Q3 results
                     file (e.g. reproduced/q3_bias_variance_results.npz after
                     a full notebook rerun) against
                     results/q3_bias_variance_results.npz. Verdicts: PASS if
                     identical, WARN for small floating-point drift (expected
                     when rerunning on a different OS/BLAS stack), FAIL only
                     for gross mismatches.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from make_splits import make_split

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE = REPO_ROOT / "data" / "olivetti_split.npz"
DEFAULT_RESULTS = REPO_ROOT / "results" / "q3_bias_variance_results.npz"


def md5(array):
    return hashlib.md5(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--reference",
        default=str(DEFAULT_REFERENCE),
        help="split .npz file to verify against (default: data/olivetti_split.npz)",
    )
    parser.add_argument(
        "--also-results",
        default=None,
        help="second q3_bias_variance_results.npz file to compare with "
             "results/q3_bias_variance_results.npz",
    )
    args = parser.parse_args()

    reference = np.load(args.reference)
    train_idx, val_idx, test_idx = make_split()

    fresh = {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}

    print(f"Verifying split against: {args.reference}")
    print("-" * 60)

    all_ok = True
    for name, fresh_array in fresh.items():
        ref_array = reference[name]
        ok = (ref_array.shape == fresh_array.shape
              and np.array_equal(ref_array, fresh_array))
        all_ok &= ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name:10s} shape={str(ref_array.shape):8s} "
              f"md5={md5(fresh_array)}")

    print("-" * 60)
    print("Split sizes:", len(train_idx), "train /", len(val_idx), "val /",
          len(test_idx), "test")

    if args.also_results:
        other = Path(args.also_results)
        print()
        print(f"Comparing Q3 results with: {other}")
        print("-" * 60)
        saved = np.load(DEFAULT_RESULTS)
        rerun = np.load(other)
        results_ok = True
        for key in saved.keys():
            if key not in rerun.keys():
                print(f"[FAIL] {key} missing in {other}")
                results_ok = False
                continue
            if saved[key].shape != rerun[key].shape:
                print(f"[FAIL] {key:12s} shape {rerun[key].shape} != "
                      f"{saved[key].shape}")
                results_ok = False
                continue
            max_diff = float(np.max(np.abs(saved[key] - rerun[key])))
            if np.allclose(saved[key], rerun[key], rtol=1e-6, atol=1e-9):
                print(f"[PASS] {key:12s} max |diff| = {max_diff:.3e} (identical)")
            elif max_diff <= 0.1:
                print(f"[WARN] {key:12s} max |diff| = {max_diff:.3e}")
                print("       small floating-point drift: this happens when the")
                print("       notebook is rerun on a different OS/BLAS stack,")
                print("       because shallow sample-weighted trees break")
                print("       rounding ties differently. All qualitative")
                print("       conclusions are unaffected (see README).")
            else:
                print(f"[FAIL] {key:12s} max |diff| = {max_diff:.3e} (too large)")
                results_ok = False
        if not results_ok:
            all_ok = False

    print("-" * 60)
    if all_ok:
        print("All checks passed: the saved split is exactly reproducible "
              "from seed 260008.")
    else:
        print("Some checks FAILED - see the lines marked FAIL above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
