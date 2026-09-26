# "v5-lite": the v5 two-stage GBDT on the E6-augmented TF-IDF candidates (artifacts\baseline_gen\), CPU only.
# No bi-encoder / cross-encoder needed, so it runs before the Kaggle results arrive (~1.5-2 h).
#
#   powershell -ExecutionPolicy Bypass -File .\run_v5lite.ps1
#   ... -Share 0.75              # share of training entities for stage 1 (default 0.72; RAM ~7 GB at 0.72)
#   ... -Xgb                     # also train the XGBoost stage-2 member (CPU; adds ~40-60 min)
#   ... -Redo stage2             # re-run one stage
# Differences from the baseline_gen submission (sub02):
#   * all 5 folds train the GBDT (no encoder folds to hold out): -Share of the entities (all their
#     candidates; v5.max_train_rows is set from it; sub02 used 50%), cross-fitted 5 ways
#   * stage 1: LightGBM lr 0.05, 127 leaves, up to 3000 trees (v5.lgb)
#   * pruning at p1 >= v5.prune_tau, then stage 2 with record / S1 competition features from stage-1 p
#   * decision: best of the (T_first, T_rest) grid and the expected-F rule, chosen on folds 1-4
# Stages: stage1, stage2, compare (vs baseline_gen = sub02, fold 0), stress (test-like density), predict.
# Outputs: artifacts\v5_lite\ (models, logs, stage*.json), subs\sub03_v5lite\. Markers: artifacts\run_v5lite\.

param(
    [double]$Share = 0.72,
    [switch]$Xgb,
    [string]$Redo = "",
    [string]$Name = "sub03_v5lite"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$env:PYTHONUNBUFFERED = "1"
$py = ".\.venv\Scripts\python.exe"
$markers = "artifacts\run_v5lite"
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

foreach ($f in @("train\India.parquet", "train\US.parquet", "test\France.parquet", "test\India.parquet", "test\US.parquet")) {
    if (-not (Test-Path "artifacts\baseline_gen\$f")) { Say "missing artifacts\baseline_gen\${f}: run run_gen.ps1 first" "Red"; exit 1 }
}

# ------------------------------------------------------------------ config: v5 on the augmented TF-IDF files
$cfgPath = (Resolve-Path ".").Path + "\$markers\pipeline.yaml"
$xgbFlag = if ($Xgb) { "True" } else { "False" }
$code = Run-Native $py @("-c", "import yaml; c = yaml.safe_load(open('configs/pipeline.yaml', encoding='utf-8')); v = c['v5']; v['dirs'] = {'union': 'baseline_gen', 'v5': 'v5_lite'}; v['gbdt_folds'] = [0, 1, 2, 3, 4]; v['cross_encoder']['enabled'] = False; v['llm']['enabled'] = False; v['xgb']['enabled'] = $xgbFlag; v['xgb']['device'] = 'cpu'; import glob, pyarrow.parquet as pq; n = sum(pq.ParquetFile(f).metadata.num_rows for f in glob.glob('artifacts/baseline_gen/train/*.parquet') if not f.endswith('_entities.parquet')); v['max_train_rows'] = int($Share * n); print('stage-1 training rows', v['max_train_rows'], 'of', n, 'train pairs'); yaml.safe_dump(c, open(r'$cfgPath', 'w', encoding='utf-8'), sort_keys=False)")
if ($code -ne 0) { Say "could not write $cfgPath" "Red"; exit 1 }
$env:BER_CONFIG = $cfgPath
Say "config: $cfgPath (union = baseline_gen, run = v5_lite, gbdt folds 0-4, xgboost $xgbFlag)" "Yellow"
Say "Missing-feature warning in stage1 is expected: the dense (Kaggle) and number-conflict columns are not in these files." "Yellow"

Stage "stage1"  @("-u", "-m", "ber.v5", "stage1")
Stage "stage2"  @("-u", "-m", "ber.v5", "stage2")
Stage "compare" @("-u", "-m", "ber.v5", "compare", "--baseline", "baseline_gen")
Stage "stress"  @("-u", "-m", "ber.gap", "stress", "--run", "v5")
Stage "predict" @("-u", "-m", "ber.v5", "predict", "--name", $Name)

Say "`nDONE. Fold-0 vs sub02: artifacts\v5_lite\compare.json (stage2_vs_baseline: ci_low > 0 -> upload)." "Green"
Say "Upload: subs\$Name\matching_results.tsv. Density stress: artifacts\experiments\C1_stress_v5.json." "Green"
