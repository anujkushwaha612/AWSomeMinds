# Runs the whole baseline on this PC, stage by stage, stopping at the first error.
# Usage (PowerShell, from the repo folder):   .\run_baseline.ps1
# Re-running after a crash resumes: finished stages are skipped automatically.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$py = ".\.venv\Scripts\python.exe"

$stages = @(
    @("-m", "pytest", "-q"),
    @("-u", "-m", "ber.folds"),
    @("-u", "-m", "ber.normalize"),
    @("-u", "-m", "ber.baseline", "build", "--split", "train"),
    @("-u", "-m", "ber.baseline", "build", "--split", "test"),
    @("-u", "-m", "ber.baseline", "train"),
    @("-u", "-m", "ber.baseline", "predict", "--name", "sub01_baseline")
)

foreach ($args_ in $stages) {
    Write-Host "`n=== python $($args_ -join ' ') ===  $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
    & $py @args_
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Stage failed (exit $LASTEXITCODE). Fix it, then run .\run_baseline.ps1 again to resume." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}
Write-Host "`nDONE. Upload output\matching_results.tsv. Metrics: artifacts\baseline\metrics.json" -ForegroundColor Green
