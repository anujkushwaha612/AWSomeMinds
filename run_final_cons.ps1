# Final submission run (E16): stage 2 with record-consensus features on the 0.980 run's stage 1.
#   fold-0 macro F0.5 0.98886 vs 0.98794 for sub_v5_noce (LB 0.980): +0.00091, 95% CI [+0.00082, +0.00101]
#
#   powershell -ExecutionPolicy Bypass -File .\run_final_cons.ps1
#
# Writes (never touches subs\sub_v5_noce):
#   subs\sub_v5_cons\          consensus stage 2, default decision              -> upload FIRST
#   subs\sub_v5_cons_fr050\    same, France-only stage-2 odds x0.5 (India/US byte-identical)
#   subs\sub_v5_cons_fr030\    same, France-only stage-2 odds x0.3 (India/US byte-identical)
# ~10-15 min, peak RAM ~8 GB: close other heavy apps. The stage-2 models + consensus matrices are already built
# (artifacts\v5\models\stage2_cons_fold*.txt, artifacts\v5\cons_test.npy).

param([double[]]$FranceOdds = @(0.5, 0.3))

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:BER_CONFIG = "$PSScriptRoot\artifacts\free\pipeline.yaml"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$env:PYTHONUNBUFFERED = "1"
$py = ".\.venv\Scripts\python.exe"
$log = "artifacts\v5\final_cons_console.txt"

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

# ---- preflight: the inputs must be the 0.980 run's test state plus the E16 artifacts
Say "=== preflight"
foreach ($f in @("artifacts\free\pipeline.yaml", "artifacts\v5\stage2_cons.json", "artifacts\v5\cons_test.npy",
                 "artifacts\v5\models\stage2_cons_fold0.txt", "artifacts\v5\models\stage2_cons_fold1.txt",
                 "artifacts\v5\models\stage2_cons_fold2.txt", "artifacts\v5\p1_test.npy")) {
    if (-not (Test-Path $f)) { Say "missing $f" "Red"; exit 1 }
}
foreach ($pair in @(@("artifacts\union\test\France.parquet", "artifacts\union\test\France.parquet.orig"),
                    @("artifacts\v5\p1_test.npy", "artifacts\v5\p1_test.npy.orig"))) {
    if ((Get-FileHash $pair[0] -Algorithm MD5).Hash -ne (Get-FileHash $pair[1] -Algorithm MD5).Hash) {
        Say "$($pair[0]) differs from $($pair[1]): the test inputs are not the 0.980 state. Stop and tell Claude." "Red"
        exit 1
    }
}
Say "preflight OK (test France union + stage-1 p1 identical to the 0.980 run)" "Green"

# ---- 1. consensus stage 2 on test -> subs\sub_v5_cons
Py @("-u", "experiments\e16_consensus.py", "predict", "--name", "sub_v5_cons", "--tag", "cons")

# ---- 2. France-only strictness variants (re-decide saved p2; India/US checked byte-identical inside)
foreach ($r in $FranceOdds) {
    $tag = "{0:000}" -f [int]([math]::Round($r * 100))
    Py @("-u", "experiments\e16_consensus.py", "fr-odds", "--tag", "cons", "--odds", "$r", "--name", "sub_v5_cons_fr$tag")
}

# ---- 3. summary: matches per S1 by country for every new submission (+ the 0.980 reference)
$names = @("sub_v5_noce", "sub_v5_cons") + ($FranceOdds | ForEach-Object { "sub_v5_cons_fr{0:000}" -f [int]([math]::Round($_ * 100)) })
$code = @"
import sys, pandas as pd
s1 = pd.read_csv('data/dataset/test/test_source1.tsv', sep='\t', dtype=str, keep_default_na=False, usecols=['entity_id', 'country'])
rows = []
for name in sys.argv[1:]:
    m = pd.read_csv(f'subs/{name}/matching_results.tsv', sep='\t', dtype=str, keep_default_na=False)
    m = s1.merge(m, left_on='entity_id', right_on='source1_entity_id', how='left')
    n = m['matched_entity_ids'].fillna('').map(lambda x: 0 if x == '' else len(x.split(',')))
    r = {'submission': name}
    for c, g in n.groupby(m['country']):
        r[c + ' matches/S1'] = round(g.mean(), 4)
        r[c + ' empty'] = round((g == 0).mean(), 4)
    rows.append(r)
print(pd.DataFrame(rows).set_index('submission').to_string())
"@
Say "`n=== summary"
$ErrorActionPreference = "Continue"
& $py -c $code @names 2>&1 | ForEach-Object { $l = "$_"; Write-Host $l; Add-Content -Path $log -Value $l -Encoding UTF8 }
$ErrorActionPreference = "Stop"
Say "`nDONE. Upload order: 1) subs\sub_v5_cons\matching_results.tsv  2) subs\sub_v5_cons_fr030\matching_results.tsv" "Green"
Say "Full console log: $log" "Green"
