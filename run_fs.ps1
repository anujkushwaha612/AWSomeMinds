# E21: unsupervised Fellegi-Sunter EM for France (India / US = sub_v5_ce, LB 0.982).
#
#   powershell -ExecutionPolicy Bypass -File .\run_fs.ps1            # validate (~8 min) + France submissions (~5 min)
#   powershell -ExecutionPolicy Bypass -File .\run_fs.ps1 -SkipValidate
#
# 1) validate: stage 2 trained on ONE country, EM adapts it label-free on the OTHER (the France situation, simulated);
#    labels are used only to score. EM-fs / EM-hyb beating "transferred" (CI > 0) = evidence it can help France.
# 2) france: EM fitted on test France -> subs\sub_v5_ce_frfs (pure EM) and subs\sub_v5_ce_frhyb (EM + supervised + CE
#    scores as fields). India / US rows are byte-identical to subs\sub_v5_ce. Never touches existing submissions.

param([switch]$SkipValidate)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:BER_CONFIG = "$PSScriptRoot\artifacts\free\pipeline.yaml"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$env:PYTHONUNBUFFERED = "1"
$py = ".\.venv\Scripts\python.exe"
$log = "artifacts\v5\run_fs_console.txt"

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

foreach ($f in @("artifacts\v5\keep_test_ce.npy", "artifacts\v5\stage2_ce.json", "artifacts\ce_kaggle\out\ce_scores_test.parquet",
                 "artifacts\v5\p2_test_consr.npy", "artifacts\v5\kept_test_consr.npy")) {
    if (-not (Test-Path $f)) { Say "missing $f" "Red"; exit 1 }
}
if (-not $SkipValidate) { Py @("-u", "experiments\e21_fs_em.py", "validate") }
Py @("-u", "experiments\e21_fs_em.py", "france")

Say "`n=== summary"
Get-Content $log | Select-String -Pattern "validate .* \(unseen\)|France EM-.*matches/S1|PASS|FAIL" | ForEach-Object { Say $_.Line "Green" }
Say "`nDONE. Upload candidates (France-only change vs the 0.982 file): subs\sub_v5_ce_frhyb\matching_results.tsv, subs\sub_v5_ce_frfs\matching_results.tsv" "Green"
