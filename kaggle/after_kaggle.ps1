# Laptop side of the free-GPU plan (KAGGLE.md), after the Kaggle runs: unpack their results and run
# the CPU stages of strategy v5 without the cross-encoder (union -> stage 1 -> stage 2 -> submission).
#
#   powershell -ExecutionPolicy Bypass -File .\kaggle\after_kaggle.ps1 -Results C:\Downloads\results_train.zip,C:\Downloads\results_dense.zip
#   powershell -ExecutionPolicy Bypass -File .\kaggle\after_kaggle.ps1            # zips already extracted
#   ... -Redo stage1                                                               # re-run one stage
# Stage markers: artifacts\free\<stage>.done (re-running skips finished stages).

param(
    [string[]]$Results = @(),
    [switch]$Xgb,
    [int]$MaxRows = 15000000,   # stage-1 training rows (RAM: ~0.25 GB per million rows at 56 features, plus keys)
    [string]$Redo = ""
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$env:PYTHONUNBUFFERED = "1"
$py = ".\.venv\Scripts\python.exe"
$markers = "artifacts\free"
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

# ------------------------------------------------------------------ unpack the Kaggle results
foreach ($z in $Results) {
    Say "extracting $z -> artifacts\"
    Expand-Archive -Path $z -DestinationPath "artifacts" -Force
}
$missing = @()
foreach ($j in @("train\US", "train\India", "test\US", "test\India", "test\France")) {
    $split, $country = $j -split "\\"
    foreach ($f in @("${country}_dense.parquet", "${country}_tfidf_cos.npy")) {
        if (-not (Test-Path "artifacts\dense\$split\$f")) { $missing += "artifacts\dense\$split\$f" }
    }
}
if ($missing.Count) {
    Say "missing dense results (run the Kaggle dense jobs for them):`n  $($missing -join "`n  ")" "Red"
    exit 1
}
if (Test-Path "artifacts\neural\eval.json") {
    Say "recall gate: artifacts\neural\eval.json (dense_R@3 vs tfidf_R@3 per country)"
}

# ------------------------------------------------------------------ config: no cross-encoder, no LLM
$env:BER_CONFIG = (Resolve-Path ".").Path + "\$markers\pipeline.yaml"
$xgbFlag = if ($Xgb) { "True" } else { "False" }   # XGBoost stage-2 member on CPU adds ~1 h; opt in with -Xgb
$code = Run-Native $py @("-c", "import yaml; c = yaml.safe_load(open('configs/pipeline.yaml', encoding='utf-8')); c['v5']['cross_encoder']['enabled'] = False; c['v5']['llm']['enabled'] = False; c['v5']['xgb']['enabled'] = $xgbFlag; c['v5']['xgb']['device'] = 'cpu'; c['v5']['max_train_rows'] = $MaxRows; yaml.safe_dump(c, open(r'$env:BER_CONFIG', 'w', encoding='utf-8'), sort_keys=False)")
if ($code -ne 0) { Say "could not write $env:BER_CONFIG" "Red"; exit 1 }

# ------------------------------------------------------------------ A5 + B + C on CPU
Stage "union_train"  @("-u", "-m", "ber.union", "--split", "train")
Stage "union_test"   @("-u", "-m", "ber.union", "--split", "test")
Stage "stage1"       @("-u", "-m", "ber.v5", "stage1")
Stage "stage2_noce"  @("-u", "-m", "ber.v5", "stage2", "--tag", "noce", "--no-ce")
Stage "predict_noce" @("-u", "-m", "ber.v5", "predict", "--tag", "noce", "--name", "sub_v5_noce")
Stage "compare"      @("-u", "-m", "ber.v5", "compare", "--baseline", "baseline_gen")
if (Test-Path "artifacts\v5_lite\per_entity_stage2.npy") {
    Stage "compare_v5lite" @("-u", "experiments\compare_runs.py", "artifacts\v5_lite\per_entity_stage2.npy", "artifacts\v5\per_entity_stage2_noce.npy")
}
Stage "stress"       @("-u", "-m", "ber.gap", "stress", "--run", "v5", "--tag", "noce")
Stage "france"       @("-u", "-m", "ber.gap", "france", "--tag", "noce")
Say "`nDONE. Submission: subs\sub_v5_noce\ (also in output\)." "Green"
Say "Upload it only if compare_v5lite above says B BETTER (vs sub03 = v5-lite); otherwise keep sub03." "Green"
