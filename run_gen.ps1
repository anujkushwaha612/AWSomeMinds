# Baseline + generator-aware features (experiments.md E6) -> a new submission, CPU only (~1-1.5 h).
#
#   powershell -ExecutionPolicy Bypass -File .\run_gen.ps1
#   powershell -ExecutionPolicy Bypass -File .\run_gen.ps1 -TrainFrac 0.5     # more training entities (more RAM)
#   ... -Redo train                                                           # re-run one stage
#   powershell -ExecutionPolicy Bypass -File .\run_gen.ps1 -TrainFrac 0.5 -Features gen
#       backup variant without CHAIN_FEATURES (reuses the feature files; own run folder / markers / sub name)
# Needs the TF-IDF baseline built (artifacts\baseline\{train,test}\*.parquet). Stages:
#   gen_train / gen_test  add GEN_FEATURES + CHAIN_FEATURES to copies of the baseline feature files -> artifacts\baseline_gen\
#   train                 5 fold LightGBMs on FEATURES + GEN_FEATURES + CHAIN_FEATURES, decision tuning, fold-0 report
#   compare               paired bootstrap vs the baseline on fold 0
#   predict               test predictions -> output\ and subs\sub02_baseline_gen\ (validated)
# LightGBM tree cap raised to -NumRounds 1500 (early stopping still decides; the baseline cap of 400 was
# reached in the E6 sample). Markers: artifacts\run_gen\<run>\<stage>.done (re-running skips finished stages).

param(
    [double]$TrainFrac = 0.3,
    [string]$Redo = "",
    [ValidateSet("genchain", "gen")][string]$Features = "genchain",
    [int]$NumRounds = 1500,
    [string]$Name = ""
)
$run = if ($Features -eq "genchain") { "baseline_gen" } else { "baseline_gen_$Features" }
if (-not $Name) { $Name = if ($Features -eq "genchain") { "sub02_baseline_gen" } else { "sub03_baseline_gen_$Features" } }

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$env:PYTHONUNBUFFERED = "1"
$py = ".\.venv\Scripts\python.exe"
$markers = "artifacts\run_gen\$run"
New-Item -ItemType Directory -Force $markers | Out-Null
$log = "$markers\console_log.txt"

function Say($msg, $color = "Cyan") {
    Write-Host $msg -ForegroundColor $color
    Add-Content -Path $log -Value $msg -Encoding UTF8
}

function Run-Native($exe, $argList) {
    $ErrorActionPreference = "Continue"
    & $exe @argList 2>&1 | ForEach-Object {
        $line = "$_"
        Write-Host $line
        Add-Content -Path $log -Value $line -Encoding UTF8
    }
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    return $code
}

function Stage($name, $argList) {
    $done = "$markers\$name.done"
    if ((Test-Path $done) -and ($Redo -ne $name)) {
        Say "--- skip $name (done $(Get-Content $done))" "DarkGray"
        return
    }
    Say "`n=== [$(Get-Date -Format 'HH:mm:ss')] $name :: python $($argList -join ' ')"
    $code = Run-Native $py $argList
    if ($code -ne 0) {
        Say "FAILED: $name (exit $code). Fix the cause, then run the same command again to resume." "Red"
        exit $code
    }
    Set-Content -Path $done -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss") -Encoding UTF8
}

foreach ($s in @("train", "test")) {
    if (-not (Get-ChildItem "artifacts\baseline\$s\*.parquet" -ErrorAction SilentlyContinue)) {
        Say "missing artifacts\baseline\$s\*.parquet: build the baseline first (run_baseline.ps1)" "Red"; exit 1
    }
}
Say "Close memory-hungry apps (browser tabs, IDEs) before training: each stage logs [mem ... free ... GB]." "Yellow"

Stage "gen_train" @("-u", "-m", "ber.gen_augment", "--split", "train")
Stage "gen_test"  @("-u", "-m", "ber.gen_augment", "--split", "test")

$env:BER_RUN = $run                # models / metrics / submission of this variant
$env:BER_FEATURES = "baseline_gen" # every variant reads the same augmented feature files
$env:BER_GEN = $Features
Stage "train"     @("-u", "-m", "ber.baseline", "train", "--train-frac", "$TrainFrac", "--num-rounds", "$NumRounds")
Stage "compare"   @("-u", "-m", "ber.baseline", "compare", "baseline", $run)
Stage "predict"   @("-u", "-m", "ber.baseline", "predict", "--name", $Name)

Say "`nDONE ($Features). Fold-0 comparison: artifacts\$run\compare_vs_baseline.json" "Green"
Say "Upload: subs\$Name\matching_results.tsv (also output\matching_results.tsv) if the verdict is KEEP." "Green"
Say "After the leaderboard score appears, write it into subs\$Name\meta.json (leaderboard_score)." "Green"
