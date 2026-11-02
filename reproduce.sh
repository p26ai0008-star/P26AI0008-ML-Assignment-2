#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproduction script for ML_assignment_P26AI0008 (Linux / macOS / Git Bash)
#
# Usage:
#   ./reproduce.sh quick    (default) verify the saved split (~1 min,
#                            downloads the Olivetti dataset once)
#   ./reproduce.sh full     verify the split, then re-execute the complete
#                            notebook in reproduced/ (~2-4 h on a laptop CPU),
#                            regenerating figures/ and both .npz outputs, and
#                            compare them with the committed copies
#
# Requirements (any of the two):
#   python -m pip install -r requirements.txt
#   conda env create -f environment.yml && conda activate ml_assignment
#
# Notes:
#   - The split indices are exactly reproducible on every platform (pure
#     integer shuffling with seed 260008).
#   - Model outputs are exactly reproducible within one environment. On a
#     different OS/BLAS stack, boosting accuracy tables can drift by a few
#     hundredths because of floating-point tie-breaking in shallow weighted
#     trees; figures and all qualitative conclusions are unaffected. See
#     README.md ("Determinism") for details.
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-quick}"

# --- pick a python interpreter ---------------------------------------------
if [ -n "${PYTHON:-}" ]; then
    PY="${PYTHON}"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
else
    PY="python"
fi

# Make sure the notebook kernel resolves to the same interpreter as $PY.
PY_BIN_DIR="$(dirname "$(${PY} -c 'import sys; print(sys.executable)')")"
case ":${PATH}:" in
    *":${PY_BIN_DIR}:"*) ;;
    *) export PATH="${PY_BIN_DIR}:${PATH}" ;;
esac

echo "Repository root : ${REPO_ROOT}"
echo "Python          : $(${PY} --version 2>&1) ($(${PY} -c 'import sys; print(sys.executable)'))"
echo "Mode            : ${MODE}"
echo

# --- check dependencies ------------------------------------------------------
if ! ${PY} -c "import numpy, sklearn, matplotlib, joblib" >/dev/null 2>&1; then
    echo "ERROR: missing Python dependencies."
    echo "Install them first with one of:"
    echo "  ${PY} -m pip install -r requirements.txt"
    echo "  conda env create -f environment.yml && conda activate ml_assignment"
    exit 1
fi

# --- quick mode: verify the saved split --------------------------------------
cd "${REPO_ROOT}"

echo "=== Step 1: regenerating the split from seed 260008 and comparing"
echo "           with the committed data/olivetti_split.npz ==="
echo
${PY} scripts/verify_splits.py
echo

if [ "${MODE}" != "full" ]; then
    echo "Quick verification finished successfully."
    echo "Run './reproduce.sh full' to re-execute the complete notebook."
    exit 0
fi

# --- full mode: re-execute the notebook --------------------------------------
OUT_DIR="${REPO_ROOT}/reproduced"
mkdir -p "${OUT_DIR}"
cp -f "${REPO_ROOT}/ML_assignment_P26AI0008.ipynb" "${OUT_DIR}/"

echo "=== Step 2: executing the full notebook in reproduced/ ==="
echo "           (all outputs are written inside reproduced/, the committed"
echo "            files in data/, results/ and figures/ are left untouched)"
echo

${PY} -m nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=-1 \
    "${OUT_DIR}/ML_assignment_P26AI0008.ipynb"

echo
echo "=== Step 3: comparing the regenerated outputs with the committed ones ==="
echo

${PY} scripts/verify_splits.py --reference "${OUT_DIR}/olivetti_split.npz"

if [ -f "${OUT_DIR}/q3_bias_variance_results.npz" ]; then
    ${PY} scripts/verify_splits.py \
        --reference "${OUT_DIR}/olivetti_split.npz" \
        --also-results "${OUT_DIR}/q3_bias_variance_results.npz"
fi

echo
echo "Regenerated figures (compare visually with the committed figures/):"
ls -lh "${OUT_DIR}/figures" 2>/dev/null || echo "  (no figures directory found)"

echo
echo "Full reproduction finished."
echo "Regenerated notebook with fresh outputs: ${OUT_DIR}/ML_assignment_P26AI0008.ipynb"
