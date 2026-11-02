# ML_assignment_P26AI0008

Machine learning assignment on **decision trees, bagging and multiclass boosting
(SAMME)** implemented from scratch on the scikit-learn **Olivetti faces** dataset
(400 grayscale images, 40 identities, 10 images each).

- **Roll number:** P26AI0008
- **Random seed used everywhere:** `260008`
- **Main code:** [`ML_assignment_P26AI0008.ipynb`](ML_assignment_P26AI0008.ipynb)
  (executed top-to-bottom; all outputs are saved inside the notebook)

## What is in this repository

```
ML_assignment_P26AI0008/
├── ML_assignment_P26AI0008.ipynb   # complete assignment code (Q1, Q2, Q3)
├── reproduce.sh                    # reproduction script (Linux / macOS / Git Bash)
├── reproduce.ps1                   # reproduction script (Windows PowerShell)
├── requirements.txt                # environment file (pip)
├── environment.yml                 # environment file (conda)
├── configs/
│   └── experiment_config.json      # seeds, grids and frozen hyperparameters
├── data/
│   └── olivetti_split.npz          # saved train/val/test split (240/80/80)
├── results/
│   └── q3_bias_variance_results.npz  # saved Q3(a) class-probability arrays
├── figures/                        # all figures produced by the notebook
│   ├── bagging_metrics.png
│   ├── bagging_face_cases.png
│   ├── adaboost_training_diagnostics.png
│   ├── adaboost_high_weight_face.png
│   └── bagging_boosting_masked_face.png
├── logs/
│   └── notebook_run.log            # final logs (all printed outputs of the run)
└── scripts/
    ├── make_splits.py              # regenerate data/olivetti_split.npz from the seed
    └── verify_splits.py            # checksum-verify the saved split
```

## Dataset

`sklearn.datasets.fetch_olivetti_faces` downloads the dataset (~6 MB) on first
use and caches it in `~/scikit_learn_data`, so the first run needs internet
access. The split is per identity: 6 train / 2 validation / 2 test images,
giving **240 / 80 / 80** images. It is generated with
`numpy.random.default_rng(260008)` and saved in `data/olivetti_split.npz`.

## Environment setup

With pip:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate ml_assignment
```

Tested with Python 3.12; numpy 1.26/2.x, scikit-learn 1.3-1.5 and
matplotlib 3.8-3.9 (the committed figures were rendered with matplotlib 3.8.2).

## Exact reproduction commands

### Quick verification (~1 minute)

Regenerates the split from the seed and verifies it against the committed
`data/olivetti_split.npz` (MD5 checksums of the three index arrays):

```bash
./reproduce.sh quick          # Linux / macOS / Git Bash
.\reproduce.ps1 quick         # Windows PowerShell
```

Manual equivalent:

```bash
python scripts/make_splits.py              # writes data/olivetti_split.npz
python scripts/verify_splits.py            # PASS/FAIL per array + checksums
```

### Full reproduction (~2-4 hours on a laptop CPU)

Re-executes the whole notebook in an isolated `reproduced/` directory (the
committed `data/`, `results/` and `figures/` files are not touched), which
regenerates `olivetti_split.npz`, `q3_bias_variance_results.npz` and all five
figures, then compares the regenerated outputs with the committed copies:

```bash
./reproduce.sh full
.\reproduce.ps1 full        # Windows
```

Manual equivalent:

```bash
mkdir reproduced
cp ML_assignment_P26AI0008.ipynb reproduced/
cd reproduced
python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=-1 \
    ML_assignment_P26AI0008.ipynb
cd ..
python scripts/verify_splits.py --reference reproduced/olivetti_split.npz \
    --also-results reproduced/q3_bias_variance_results.npz
```

(The reproduction scripts additionally put the chosen Python interpreter first
on `PATH` before launching the kernel, so the notebook always runs with the
environment you installed the requirements into.)

## Key results (from `logs/notebook_run.log`)

| Model | Clean test acc. | Masked test acc. | Bias² | Variance |
|---|---|---|---|---|
| Single tree (max_depth=None, min_samples_leaf=1) | 0.5875 | 0.4125 | 0.3730 | 0.6403 |
| Bagging (B=100, majority vote) | 0.8500 | 0.6500 | 0.4918 | 0.0245 |
| SAMME boosting (depth 4, 25 rounds) | 0.2875 | 0.2000 | 0.8855 | 0.0307 |

The masked test set applies a 16x16 black square (rows/cols 24:40) over the
centre of every test image. Selected hyperparameters were chosen on the
validation set; see `configs/experiment_config.json` for the full grids and
`logs/notebook_run.log` for every printed table.

## Determinism

All randomness flows from the seed `260008`: the dataset split, the bootstrap
resamples (`default_rng(260008)`) and the per-tree `random_state` values.
Verified behaviour of a rerun:

- **The split is exactly reproducible everywhere** (pure integer index
  shuffling, independent of platform or library versions). This is what
  `./reproduce.sh quick` checks.
- **The single tree and bagging results (Q1) reproduce exactly** on Linux with
  scikit-learn 1.3/1.4/1.5 and numpy 1.26/2.1 - every accuracy, table entry
  and vote count matches, and all figures regenerate pixel-identical.
- **The boosting tables (Q2/Q3) reproduce exactly within one environment.**
  Across a different OS/BLAS stack the boosting validation/test accuracies can
  drift by a few hundredths, because shallow sample-weighted decision trees
  break floating-point ties differently (the training trajectories stay
  indistinguishable - the diagnostic figures regenerate pixel-identical - and
  every qualitative conclusion is unchanged). `verify_splits.py` reports this
  case as `WARN` rather than `FAIL`.

If you need bit-exact boosting numbers, rerun in the same environment/OS the
submission was produced with; the committed outputs themselves are the
reference results.
