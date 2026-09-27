# E20 cross-encoder, round 2 (laptop side). Round 1 (sub_v5_ce): fold-0 0.99048 vs consr 0.98875, +0.00173 CI [+0.00162, +0.00185].
#
#   1) powershell -ExecutionPolicy Bypass -File .\run_ce.ps1 -Step prep
#        -> artifacts\ce_kaggle2\  (new ce_train.parquet: wider band, more pairs; same ce_score.parquet; ce_kaggle.py)
#        upload that folder as a new Kaggle dataset (or a new version of ce-data), run the notebook with the cell printed below
#   2) powershell -ExecutionPolicy Bypass -File .\run_ce.ps1 -Step combine -Scores <folder with the 3 ce_scores_*.parquet>
#        -> subs\sub_v5_ce2\ and subs\sub_v5_ce2_fronly\ + fold-0 comparison with consr AND with round 1
# Never touches subs\sub_v5_consr or subs\sub_v5_ce.

param(
    [Parameter(Mandatory = $true)][ValidateSet("prep", "combine")][string]$Step,
    [string]$Scores = "",
    [string]$Tag = "ce2",
    [string]$Name = "sub_v5_ce2",
    [double]$TrainLo = 0.005,
    [double]$TrainHi = 0.995,
    [int]$NTrain = 700000
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:BER_CONFIG = "$PSScriptRoot\artifacts\free\pipeline.yaml"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$env:PYTHONUNBUFFERED = "1"
$py = ".\.venv\Scripts\python.exe"
$log = "artifacts\v5\run_ce_console.txt"

function Say($msg, $color = "Cyan") {
    Write-Host $msg -ForegroundColor $color
    Add-Content -Path $log -Value $msg -Encoding UTF8
}

function Py($argList) {
    Say "`n=== [$(Get-Date -Format 'HH:mm:ss')] python $($argList -join ' ')"
    $ErrorActionPreference = "Continue"
    & $py @argList 2>&1 | ForEach-Object { $l = "$_"; Write-Host $l; Add-Content -Path $log -Value $l -Encoding UTF8 }
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($code -ne 0) { Say "FAILED (exit $code): python $($argList -join ' ')" "Red"; exit $code }
}

if ($Step -eq "prep") {
    if (-not (Test-Path "artifacts\ce_kaggle\ce_score.parquet")) { Say "missing artifacts\ce_kaggle\ce_score.parquet (round 1)" "Red"; exit 1 }
    Py @("-u", "experiments\e20_ce_prep.py", "--train-lo", "$TrainLo", "--train-hi", "$TrainHi", "--n-train", "$NTrain",
         "--out", "ce_kaggle2", "--train-only")
    Copy-Item "artifacts\ce_kaggle\ce_score.parquet" "artifacts\ce_kaggle2\ce_score.parquet" -Force
    Copy-Item "kaggle\ce_kaggle.py" "artifacts\ce_kaggle2\ce_kaggle.py" -Force
    Get-ChildItem "artifacts\ce_kaggle2" | ForEach-Object { Say ("  {0,-20} {1,8:N0} MB" -f $_.Name, ($_.Length / 1MB)) "Green" }
    Say "`nUpload artifacts\ce_kaggle2\ as a Kaggle dataset (GPU T4 x2, Internet on). Notebook cells:" "Green"
    Say "  import glob, shutil; shutil.copy(sorted(glob.glob('/kaggle/input/**/ce_kaggle.py', recursive=True))[-1], '/kaggle/working/ce_kaggle.py')" "Yellow"
    Say "  !cd /kaggle/working && CE_EPOCHS=2 CE_TRAIN_MIN=60 CE_BS=128 python -u ce_kaggle.py 2>&1 | grep -v -i warning" "Yellow"
    Say "Attach ONLY the round-2 dataset (both have files with the same names)." "Yellow"
    exit 0
}

# ---- combine
if (-not $Scores) { Say "pass -Scores <folder with ce_scores_f0/f12/test.parquet>" "Red"; exit 1 }
foreach ($p in @("f0", "f12", "test")) {
    if (-not (Get-ChildItem -Path $Scores -Recurse -Filter "ce_scores_$p.parquet" -ErrorAction SilentlyContinue)) {
        Say "missing ce_scores_$p.parquet under $Scores" "Red"; exit 1
    }
}
Py @("-u", "experiments\e20_ce_combine.py", "--scores", $Scores, "--name", $Name, "--tag", $Tag)
if (Test-Path "artifacts\v5\per_entity_stage2_ce.npy") {
    Say "`n=== round 2 ($Tag) vs round 1 (ce) on fold 0"
    Py @("experiments\compare_runs.py", "artifacts\v5\per_entity_stage2_ce.npy", "artifacts\v5\per_entity_stage2_$Tag.npy")
}
Say "`nDONE. Candidate upload: subs\$Name\matching_results.tsv (France-only variant: subs\${Name}_fronly)" "Green"
Say "Full console log: $log" "Green"
