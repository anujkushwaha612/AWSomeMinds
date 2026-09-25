# E3: retrain on ALL training entities (baseline used 30%), reusing the baseline's features.
# Prints the new fold-0 macro F0.5 and a paired-bootstrap comparison against the baseline.
# Builds a submission (sub02_e3_all) only if the comparison says the new model is better.
#
# Usage (PowerShell, from the repo folder):   powershell -ExecutionPolicy Bypass -File .\run_e3.ps1
# Nothing of the baseline is overwritten: models/metrics go to artifacts\e3_all\.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$env:BER_RUN = "e3_all"          # where this run's models / OOF / metrics go
$env:BER_FEATURES = "baseline"   # reuse the features already built by the baseline run
$py = ".\.venv\Scripts\python.exe"

function Run($argList) {
    Write-Host "`n=== python $($argList -join ' ') ===  $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
    & $py @argList
    if ($LASTEXITCODE -ne 0) { Write-Host "Failed (exit $LASTEXITCODE)." -ForegroundColor Red; exit $LASTEXITCODE }
}

Run @("-u", "-m", "ber.baseline", "train", "--train-frac", "1.0")
Run @("-u", "-m", "ber.baseline", "compare", "baseline", "e3_all")

$cmp = Get-Content "artifacts\e3_all\compare_vs_baseline.json" -Raw | ConvertFrom-Json
Write-Host "`nBaseline fold-0 F0.5: $($cmp.a.macro_f05)   E3 fold-0 F0.5: $($cmp.b.macro_f05)" -ForegroundColor Yellow
Write-Host "Delta: $($cmp.delta.delta)  CI [$($cmp.delta.ci_low), $($cmp.delta.ci_high)]  -> $($cmp.verdict)" -ForegroundColor Yellow

if ($cmp.verdict.StartsWith("KEEP")) {
    Run @("-u", "-m", "ber.baseline", "predict", "--name", "sub02_e3_all")
    Write-Host "`nDONE. Upload output\matching_results.tsv (snapshot: subs\sub02_e3_all)." -ForegroundColor Green
} else {
    Write-Host "`nNo submission built: E3 is not clearly better than the baseline." -ForegroundColor Green
}
