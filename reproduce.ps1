# ---------------------------------------------------------------------------
# Reproduction script for ML_assignment_P26AI0008 (Windows PowerShell)
#
# Usage:
#   .\reproduce.ps1 quick    (default) verify the saved split (~1 min,
#                            downloads the Olivetti dataset once)
#   .\reproduce.ps1 full     verify the split, then re-execute the complete
#                            notebook in reproduced\ (~2-4 h on a laptop CPU),
#                            regenerating figures\ and both .npz outputs, and
#                            compare them with the committed copies
#
# Requirements (any of the two):
#   python -m pip install -r requirements.txt
#   conda env create -f environment.yml; conda activate ml_assignment
#
# If script execution is disabled on your machine, run instead:
#   powershell -ExecutionPolicy Bypass -File .\reproduce.ps1 quick
# ---------------------------------------------------------------------------

$RepoRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$Mode = if ($args.Count -gt 0) { $args[0] } else { "quick" }

function Invoke-Step {
    # Run a python script and stop the reproduction if it fails.
    # Out-Host keeps the script output on the display but out of this
    # function's return value, so the caller only receives $true/$false.
    param([string[]]$Command)
    & @Command | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Reproduction stopped: the command above reported an error."
        return $false
    }
    return $true
}

function Main {
    # --- pick a python interpreter ------------------------------------------
    $Py = if ($env:PYTHON) { $env:PYTHON }
          elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" }
          else { "python3" }

    # Make sure the notebook kernel resolves to the same interpreter as $Py.
    $PyExe = & $Py -c 'import sys; print(sys.executable)'
    $PyBinDir = Split-Path -Parent $PyExe
    if ($PyBinDir -and ($env:PATH -notlike "*$PyBinDir*")) {
        $env:PATH = "$PyBinDir;$env:PATH"
    }

    Write-Host "Repository root : $RepoRoot"
    Write-Host "Python          : $(& $Py --version) ($PyExe)"
    Write-Host "Mode            : $Mode"
    Write-Host ""

    # --- check dependencies --------------------------------------------------
    & $Py -c "import numpy, sklearn, matplotlib, joblib" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: missing Python dependencies."
        Write-Host "Install them first with one of:"
        Write-Host "  $Py -m pip install -r requirements.txt"
        Write-Host "  conda env create -f environment.yml; conda activate ml_assignment"
        return
    }

    Set-Location $RepoRoot

    # --- quick mode: verify the saved split ----------------------------------
    Write-Host "=== Step 1: regenerating the split from seed 260008 and comparing"
    Write-Host "           with the committed data\olivetti_split.npz ==="
    Write-Host ""
    if (-not (Invoke-Step @($Py, "scripts\verify_splits.py"))) { return }
    Write-Host ""

    if ($Mode -ne "full") {
        Write-Host "Quick verification finished successfully."
        Write-Host "Run '.\reproduce.ps1 full' to re-execute the complete notebook."
        return
    }

    # --- full mode: re-execute the notebook ----------------------------------
    $OutDir = Join-Path $RepoRoot "reproduced"
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    Copy-Item -Force (Join-Path $RepoRoot "ML_assignment_P26AI0008.ipynb") $OutDir

    Write-Host "=== Step 2: executing the full notebook in reproduced\ ==="
    Write-Host "           (all outputs are written inside reproduced\, the committed"
    Write-Host "            files in data\, results\ and figures\ are left untouched)"
    Write-Host ""

    $NbPath = Join-Path $OutDir "ML_assignment_P26AI0008.ipynb"
    $NbArgs = @(
        "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.kernel_name=python3",
        "--ExecutePreprocessor.timeout=-1",
        $NbPath
    )
    $FullCmd = @($Py) + $NbArgs
    if (-not (Invoke-Step $FullCmd)) { return }

    Write-Host ""
    Write-Host "=== Step 3: comparing the regenerated outputs with the committed ones ==="
    Write-Host ""

    if (-not (Invoke-Step @($Py, "scripts\verify_splits.py", "--reference", (Join-Path $OutDir "olivetti_split.npz")))) { return }

    $Q3File = Join-Path $OutDir "q3_bias_variance_results.npz"
    if (Test-Path $Q3File) {
        if (-not (Invoke-Step @($Py, "scripts\verify_splits.py", "--reference", (Join-Path $OutDir "olivetti_split.npz"), "--also-results", $Q3File))) { return }
    }

    Write-Host ""
    Write-Host "Regenerated figures (compare visually with the committed figures\):"
    $FigDir = Join-Path $OutDir "figures"
    if (Test-Path $FigDir) {
        Get-ChildItem $FigDir | Format-Table Name, Length -AutoSize
    } else {
        Write-Host "  (no figures directory found)"
    }

    Write-Host ""
    Write-Host "Full reproduction finished."
    Write-Host "Regenerated notebook with fresh outputs: $NbPath"
}

Main
