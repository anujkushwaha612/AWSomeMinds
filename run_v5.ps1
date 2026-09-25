# Strategy v5 (A + B + C) end to end on the GPU machine. See GPU_RUNBOOK.md first.
# Usage (PowerShell, repo folder):   powershell -ExecutionPolicy Bypass -File .\run_v5.ps1
#   -CpuOnly   skip the GPU stages A0-A4 (they ran on Colab / the GPU box; artifacts\neural and
#              artifacts\dense were copied here) and run A5 -> B -> C on this machine.
# Stops at the first failing stage; re-running resumes (finished stages are skipped or
# restart from their checkpoint).
param([switch]$CpuOnly)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$py = ".\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force "artifacts\v5" | Out-Null
$logFile = "artifacts\v5\run_v5_console.txt"

function Stage($title, $argList) {
    $stamp = Get-Date -Format "HH:mm:ss"
    Write-Host "`n=== [$stamp] $title :: python $($argList -join ' ')" -ForegroundColor Cyan
    "`n=== [$stamp] $title :: python $($argList -join ' ')" | Out-File -Append -Encoding utf8 $logFile
    # PowerShell 5.1 turns native stderr lines into errors; treat them as text, judge by exit code
    $ErrorActionPreference = "Continue"
    & $py @argList 2>&1 | ForEach-Object { "$_" } | Tee-Object -Append -FilePath $logFile
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($code -ne 0) {
        Write-Host "FAILED: $title (exit $code). Fix and re-run .\run_v5.ps1 to resume." -ForegroundColor Red
        exit $code
    }
}

# prerequisites: data in data\dataset, artifacts\folds.parquet, artifacts\norm, artifacts\baseline
if (-not (Test-Path "artifacts\baseline\train\US.parquet")) {
    Write-Host "artifacts\baseline is missing: copy it from the laptop or run .\run_baseline.ps1 first." -ForegroundColor Red
    exit 1
}

Stage "tests"                         @("-m", "pytest", "-q")
if ($CpuOnly) {
    foreach ($f in @("artifacts\neural\eval.json", "artifacts\dense\train\US_dense.parquet",
                     "artifacts\dense\test\France_dense.parquet")) {
        if (-not (Test-Path $f)) { Write-Host "-CpuOnly needs $f (copy artifacts\neural and artifacts\dense from the GPU run)." -ForegroundColor Red; exit 1 }
    }
    Write-Host "CpuOnly: skipping A0-A4 (GPU stages)" -ForegroundColor Yellow
} else {
    Stage "A0 environment + throughput"   @("-u", "-m", "ber.neural.env_check")
    Stage "A1 training pairs (CPU)"       @("-u", "-m", "ber.neural.pairs")
    Stage "A2-A3 fine-tune + recall gate" @("-u", "-m", "ber.neural.train_biencoder")
    Stage "A4 dense search: train"        @("-u", "-m", "ber.neural.dense_retrieve", "--split", "train")
    Stage "A4 dense search: test"         @("-u", "-m", "ber.neural.dense_retrieve", "--split", "test")
}
Stage "A5 union features: train"      @("-u", "-m", "ber.union", "--split", "train")
Stage "A5 union features: test"       @("-u", "-m", "ber.union", "--split", "test")
Stage "B stage 1 (union GBDT)"        @("-u", "-m", "ber.v5", "stage1")
Stage "B stage 2 + decision"          @("-u", "-m", "ber.v5", "stage2")
Stage "B compare vs baseline"         @("-u", "-m", "ber.v5", "compare")
Stage "C1 density stress"             @("-u", "-m", "ber.gap", "stress", "--run", "v5")
Stage "B predict test"                @("-u", "-m", "ber.v5", "predict", "--name", "sub03_v5")
Stage "C2 France diagnostics"         @("-u", "-m", "ber.gap", "france")

Write-Host "`nDONE. Submission: output\matching_results.tsv (snapshot subs\sub03_v5)." -ForegroundColor Green
Write-Host "Send back: artifacts\v5\log.txt, artifacts\neural\eval.json, artifacts\v5\*.json, artifacts\experiments\C*.json" -ForegroundColor Green
