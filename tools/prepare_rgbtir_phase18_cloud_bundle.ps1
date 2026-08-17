param(
    [string]$OutputZip = "D:\12525\Documents\pytorch\aic_rgbtir_phase18_d1_cloud_bundle_20260814.zip"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RecoveryRoot = Join-Path $RepoRoot "outputs\aic_rgbtir_phase16_recovery_returned_20260813\extracted\outputs\aic_rgbtir_phase16_recovery_v1"
$StageRoot = Join-Path (Split-Path -Parent $OutputZip) "aic_rgbtir_phase18_bundle"
$ExpectedBankSha = "4A429B24353171CA1A887059A55D5597EF73A6E94E87B1C00D8582515E1567DD"

if (-not (Test-Path -LiteralPath $RecoveryRoot -PathType Container)) {
    throw "Recovered Phase 1.6 assets are missing: $RecoveryRoot"
}
if (Test-Path -LiteralPath $StageRoot) {
    $resolvedStage = (Resolve-Path -LiteralPath $StageRoot).Path
    $resolvedParent = (Resolve-Path -LiteralPath (Split-Path -Parent $StageRoot)).Path
    if (-not $resolvedStage.StartsWith($resolvedParent, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe staging path: $resolvedStage"
    }
    Remove-Item -LiteralPath $resolvedStage -Recurse -Force
}
New-Item -ItemType Directory -Path $StageRoot | Out-Null

$dirs = @(
    "src\aic_rgbtir",
    "tools\cloud",
    "configs\phase18_candidates",
    "tests",
    "assets\manifests",
    "assets\teacher_bank",
    "assets\c2_reference",
    "reports"
)
foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Path (Join-Path $StageRoot $dir) -Force | Out-Null
}

Get-ChildItem -LiteralPath (Join-Path $RepoRoot "src\aic_rgbtir") -File -Filter "*.py" | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $StageRoot "src\aic_rgbtir")
}
Copy-Item -LiteralPath (Join-Path $RepoRoot "tools\run_rgbtir_phase16.py") -Destination (Join-Path $StageRoot "tools")
Copy-Item -LiteralPath (Join-Path $RepoRoot "tools\run_rgbtir_phase18.py") -Destination (Join-Path $StageRoot "tools")
Copy-Item -LiteralPath (Join-Path $RepoRoot "tools\download_exact_asset.py") -Destination (Join-Path $StageRoot "tools")
Copy-Item -LiteralPath (Join-Path $RepoRoot "tools\cloud\install_rgbtir_phase18_env.sh") -Destination (Join-Path $StageRoot "tools\cloud")
Copy-Item -LiteralPath (Join-Path $RepoRoot "tools\cloud\run_rgbtir_phase18_probe.sh") -Destination (Join-Path $StageRoot "tools\cloud")
Copy-Item -LiteralPath (Join-Path $RepoRoot "configs\aic_rgbtir_phase18.cloud.example.yaml") -Destination (Join-Path $StageRoot "configs")
Copy-Item -Path (Join-Path $RepoRoot "configs\phase18_candidates\*.yaml") -Destination (Join-Path $StageRoot "configs\phase18_candidates")
Copy-Item -LiteralPath (Join-Path $RepoRoot "requirements-rgbtir-phase1.txt") -Destination $StageRoot
Copy-Item -LiteralPath (Join-Path $RepoRoot "tests\test_rgbtir_phase16.py") -Destination (Join-Path $StageRoot "tests")
Copy-Item -LiteralPath (Join-Path $RepoRoot "tests\test_rgbtir_phase18.py") -Destination (Join-Path $StageRoot "tests")
Copy-Item -LiteralPath (Join-Path $RepoRoot "tests\test_rgbtir_phase18_bundle_contract.py") -Destination (Join-Path $StageRoot "tests")
Copy-Item -LiteralPath (Join-Path $RepoRoot "reports\aic_rgbtir_phase18_cloud_runbook_2026_08_14.md") -Destination (Join-Path $StageRoot "reports")

$manifestNames = @("repair_probe_train.jsonl", "repair_dev.jsonl", "repair_full_train.jsonl", "official_val.jsonl", "split_summary.json")
foreach ($name in $manifestNames) {
    Copy-Item -LiteralPath (Join-Path $RecoveryRoot "manifests\$name") -Destination (Join-Path $StageRoot "assets\manifests")
}
Copy-Item -LiteralPath (Join-Path $RecoveryRoot "rgb_teacher_bank\teacher_bank.pt") -Destination (Join-Path $StageRoot "assets\teacher_bank")
Copy-Item -LiteralPath (Join-Path $RecoveryRoot "rgb_teacher_bank\summary.json") -Destination (Join-Path $StageRoot "assets\teacher_bank")
Copy-Item -LiteralPath (Join-Path $RecoveryRoot "probe_candidates\C2\summary.json") -Destination (Join-Path $StageRoot "assets\c2_reference\c2_probe_summary.json")
Copy-Item -LiteralPath (Join-Path $RecoveryRoot "recovery_diagnostics\summary.json") -Destination (Join-Path $StageRoot "assets\c2_reference\c2_reference.json")

$bankPath = Join-Path $StageRoot "assets\teacher_bank\teacher_bank.pt"
$actualBankSha = (Get-FileHash -LiteralPath $bankPath -Algorithm SHA256).Hash.ToUpperInvariant()
if ($actualBankSha -ne $ExpectedBankSha) {
    throw "Teacher Bank hash mismatch: $actualBankSha"
}

$entries = @{}
Get-ChildItem -LiteralPath $StageRoot -Recurse -File | Sort-Object FullName | ForEach-Object {
    $relative = $_.FullName.Substring($StageRoot.Length + 1).Replace("\", "/")
    $entries[$relative] = @{
        bytes = $_.Length
        sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToUpperInvariant()
    }
}
$bundleManifest = @{
    schema_version = 1
    phase = "1.8A"
    scope = "D1_RETENTION_PROBE_ONLY"
    generated_at = "2026-08-14T00:00:00Z"
    teacher_bank_sha256 = $actualBankSha
    files = $entries
}
$bundleManifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $StageRoot "BUNDLE_MANIFEST.json") -Encoding utf8

$zipParent = Split-Path -Parent $OutputZip
if (-not (Test-Path -LiteralPath $zipParent)) {
    New-Item -ItemType Directory -Path $zipParent -Force | Out-Null
}
if (Test-Path -LiteralPath $OutputZip) {
    Remove-Item -LiteralPath $OutputZip -Force
}
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::Open(
    $OutputZip,
    [System.IO.Compression.ZipArchiveMode]::Create
)
try {
    Get-ChildItem -LiteralPath $StageRoot -Recurse -File | Sort-Object FullName | ForEach-Object {
        $relative = $_.FullName.Substring($StageRoot.Length + 1).Replace("\", "/")
        $entryName = "aic_rgbtir_phase18_bundle/$relative"
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $archive,
            $_.FullName,
            $entryName,
            [System.IO.Compression.CompressionLevel]::Optimal
        ) | Out-Null
    }
}
finally {
    $archive.Dispose()
}

$result = @{
    status = "PHASE_18_CLOUD_BUNDLE_READY"
    staging_root = $StageRoot
    zip = $OutputZip
    bytes = (Get-Item -LiteralPath $OutputZip).Length
    sha256 = (Get-FileHash -LiteralPath $OutputZip -Algorithm SHA256).Hash.ToUpperInvariant()
    teacher_bank_sha256 = $actualBankSha
}
$result | ConvertTo-Json -Depth 4
