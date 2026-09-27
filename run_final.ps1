# Final run: v5 stages on the TF-IDF + dense union with ALL folds-0-2 training rows for stage 1 (streamed, no RAM
# cap), optional larger stage-2 LightGBM, the France-only vocabulary / address features (union test France rebuilt
# by experiments/e15_france_repredict.py), then a per-entity comparison with the 0.980 run and a submission.
#
#   powershell -ExecutionPolicy Bypass -File .\run_final.ps1                 # stage 2 with the default parameters
#   powershell -ExecutionPolicy Bypass -File .\run_final.ps1 -Stage2Big      # stage 2: 255 leaves, lr 0.03, min leaf 100
#   ... -Redo stage2                                                          # re-run one stage
# Needs the after_kaggle run (artifacts\union\, artifacts\free\pipeline.yaml). Outputs: artifacts\v5_final\,
# subs\sub_v5_final\. Markers: artifacts\run_final\ (re-running skips finished stages).

param(
    [switch]$Stage2Big,
    [string]$Redo = "",
    [string]$Name = "sub_v5_final"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$env:PYTHONUNBUFFERED = "1"
$py = ".\.venv\Scripts\python.exe"
$markers = "artifacts\run_final"
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

if (-not (Test-Path "artifacts\free\pipeline.yaml")) { Say "missing artifacts\free\pipeline.yaml: run kaggle\after_kaggle.ps1 first" "Red"; exit 1 }
if (-not (Test-Path "artifacts\union\test\France.parquet.orig")) {
    Say "France features not rebuilt yet: run experiments\e15_france_repredict.py first (it keeps the old file as .orig)" "Yellow"
}
Say "Close memory-hungry apps: stage 1 trains on all folds-0-2 rows (~35M), peak RAM ~8-9 GB." "Yellow"

$cfgPath = (Resolve-Path ".").Path + "\$markers\pipeline.yaml"
$big = if ($Stage2Big) { "True" } else { "False" }
$code = Run-Native $py @("-c", "import yaml; c = yaml.safe_load(open(r'artifacts\free\pipeline.yaml', encoding='utf-8')); v = c['v5']; v['dirs'] = dict(v.get('dirs') or {}, v5='v5_final'); v['stage1_stream'] = True; v['max_train_rows'] = 10**9; v['lgb_stage2'] = {'num_leaves': 255, 'learning_rate': 0.03, 'min_data_in_leaf': 100} if $big else {}; yaml.safe_dump(c, open(r'$cfgPath', 'w', encoding='utf-8'), sort_keys=False); print('stage1 streamed on all folds-0-2 rows | stage2 overrides:', v['lgb_stage2'])")
if ($code -ne 0) { Say "could not write $cfgPath" "Red"; exit 1 }
$env:BER_CONFIG = $cfgPath

Stage "stage1"       @("-u", "-m", "ber.v5", "stage1")
Stage "stage2_noce"  @("-u", "-m", "ber.v5", "stage2", "--tag", "noce", "--no-ce")
Stage "compare_v5"   @("-u", "experiments\compare_runs.py", "artifacts\v5\per_entity_stage2_noce.npy", "artifacts\v5_final\per_entity_stage2_noce.npy")
Stage "predict_noce" @("-u", "-m", "ber.v5", "predict", "--tag", "noce", "--name", $Name)

Say "`nDONE. compare_v5 above: B = this run vs A = the 0.980 run (fold-0, India + US)." "Green"
Say "Upload subs\$Name\matching_results.tsv if compare_v5 says B BETTER." "Green"
