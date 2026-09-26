# One command, fresh clone -> submission: TF-IDF baseline + strategy v5 (A bi-encoder, B union GBDT, C gap).
#
# Usage (Windows PowerShell, repo folder, data in data\dataset\{train,test}\):
#   powershell -ExecutionPolicy Bypass -File .\run_all.ps1
# Options:
#   -Cuda cu124        PyTorch CUDA wheel tag (cu121 / cu124 / cu126 / cu128 - match `nvidia-smi`)
#   -SkipSetup         do not create the venv / install packages
#   -Redo <stage>      re-run one stage (and nothing else is forced), e.g. -Redo stage2
#   -AllowCpu          smoke tests only: allow running the GPU stages without CUDA
#   -Limit <n>         smoke tests only: rows per source per country for the baseline build
#   -Tag <name>        marker folder artifacts\<name> (default run_all; use another for smoke tests)
# Every stage writes artifacts\run_all\<stage>.done when it succeeds; re-running the script
# skips finished stages, so after a crash just run the same command again.

param(
    [string]$Cuda = "cu124",
    [switch]$SkipSetup,
    [string]$Redo = "",
    [switch]$AllowCpu,
    [int]$Limit = 0,
    [string]$Tag = "run_all"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$env:PYTHONUNBUFFERED = "1"
$markers = "artifacts\$Tag"
New-Item -ItemType Directory -Force $markers | Out-Null
$log = "$markers\console_log.txt"

function Say($msg, $color = "Cyan") {
    Write-Host $msg -ForegroundColor $color
    Add-Content -Path $log -Value $msg -Encoding UTF8
}

function Run-Native($exe, $argList) {
    # Windows PowerShell 5.1 turns native stderr lines into errors: treat them as text, judge by exit code
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

# ------------------------------------------------------------------ data check
foreach ($f in @("train\train_source1.tsv", "train\train_source2.tsv", "train\train_source3.tsv",
                 "train\train_ground_truth.tsv", "test\test_source1.tsv", "test\test_source2.tsv",
                 "test\test_source3.tsv")) {
    if (-not (Test-Path "data\dataset\$f")) { Say "missing data\dataset\$f" "Red"; exit 1 }
}

# ------------------------------------------------------------------ setup
$py = ".\.venv\Scripts\python.exe"
if (-not $SkipSetup) {
    if (-not (Test-Path $py)) {
        Say "=== creating .venv"
        if ((Run-Native "python" @("-m", "venv", ".venv")) -ne 0) { Say "python -m venv failed" "Red"; exit 1 }
    }
    Say "=== installing packages (torch $Cuda)"
    foreach ($a in @(@("-m", "pip", "install", "-q", "--upgrade", "pip"),
                     @("-m", "pip", "install", "-q", "-r", "requirements.txt"),
                     @("-m", "pip", "install", "-q", "torch", "--index-url", "https://download.pytorch.org/whl/$Cuda"),
                     @("-m", "pip", "install", "-q", "-r", "requirements-gpu.txt"),
                     @("-m", "pip", "install", "-q", "-e", "."))) {
        if ((Run-Native $py $a) -ne 0) { Say "pip failed: $($a -join ' ')" "Red"; exit 1 }
    }
}
if ((Run-Native $py @("-c", "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')")) -ne 0) {
    Say "PyTorch does not import. If Windows says 'An Application Control policy has blocked this file', Smart App Control must be off." "Red"
    exit 1
}

$limitArgs = @()
if ($Limit -gt 0) { $limitArgs = @("--limit", "$Limit") }
$cpuArgs = @()
if ($AllowCpu) { $cpuArgs = @("--allow-cpu") }

# ------------------------------------------------------------------ baseline (TF-IDF candidates, CPU)
Stage "tests"            @("-m", "pytest", "-q")
Stage "folds"            @("-u", "-m", "ber.folds")
Stage "normalize"        @("-u", "-m", "ber.normalize")
Stage "baseline_build_train" (@("-u", "-m", "ber.baseline", "build", "--split", "train") + $limitArgs)
Stage "baseline_build_test"  (@("-u", "-m", "ber.baseline", "build", "--split", "test") + $limitArgs)
Stage "baseline_train"   @("-u", "-m", "ber.baseline", "train")

# ------------------------------------------------------------------ A: bi-encoder (GPU)
Stage "env_check"        (@("-u", "-m", "ber.neural.env_check") + $cpuArgs)
Stage "pairs"            @("-u", "-m", "ber.neural.pairs")
Stage "train_encoder"    @("-u", "-m", "ber.neural.train_biencoder")
Stage "dense_train"      @("-u", "-m", "ber.neural.dense_retrieve", "--split", "train")
Stage "dense_test"       @("-u", "-m", "ber.neural.dense_retrieve", "--split", "test")

# ------------------------------------------------------------------ A5 + B: union features, GBDT (CPU)
Stage "union_train"      @("-u", "-m", "ber.union", "--split", "train")
Stage "union_test"       @("-u", "-m", "ber.union", "--split", "test")
Stage "stage1"           @("-u", "-m", "ber.v5", "stage1")
# safe submission first (no cross-encoder); also the ablation reference for the CE
Stage "stage2_noce"      @("-u", "-m", "ber.v5", "stage2", "--tag", "noce", "--no-ce")
Stage "predict_noce"     @("-u", "-m", "ber.v5", "predict", "--tag", "noce", "--name", "sub_v5_noce")

# ------------------------------------------------------------------ D: cross-encoder on the gray zone (GPU)
Stage "ce_pairs"         @("-u", "-m", "ber.neural.cross_encoder", "pairs")
Stage "ce_train"         (@("-u", "-m", "ber.neural.cross_encoder", "train") + $cpuArgs)
Stage "ce_score_train"   (@("-u", "-m", "ber.neural.cross_encoder", "score", "--split", "train") + $cpuArgs)
Stage "ce_score_test"    (@("-u", "-m", "ber.neural.cross_encoder", "score", "--split", "test") + $cpuArgs)
Stage "stage2"           @("-u", "-m", "ber.v5", "stage2")
Stage "compare"          @("-u", "-m", "ber.v5", "compare")

# ------------------------------------------------------------------ C + submission
Stage "stress"           @("-u", "-m", "ber.gap", "stress", "--run", "v5")
Stage "predict"          @("-u", "-m", "ber.v5", "predict", "--name", "sub_v5")
Stage "france"           @("-u", "-m", "ber.gap", "france")

Say "`nDONE. Two snapshots: subs\sub_v5_noce (no cross-encoder) and subs\sub_v5 (with; now in output\)." "Green"
Say "Pick by artifacts\v5\compare.json -> ce_ablation_stage2_vs_stage2_noce (upload sub_v5 only if its CI is > 0)." "Green"
Say "Offline results: artifacts\v5\stage2.json, artifacts\v5\compare.json, artifacts\neural\eval.json, artifacts\experiments\C*.json" "Green"
