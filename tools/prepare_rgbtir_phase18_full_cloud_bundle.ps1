param(
    [string]$OutputZip = "D:\12525\Documents\pytorch\aic_rgbtir_phase18r_5090_cloud_bundle_20260815.zip"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$StageParent = (Split-Path -Parent $OutputZip)
$StageRoot = Join-Path $StageParent "aic_rgbtir_phase18_full_bundle"
$ManifestRoot = Join-Path $RepoRoot "outputs\aic_rgbtir_phase16_v1\manifests"
$TrainClean = Join-Path $RepoRoot "outputs\aic_rgbtir_phase0_v1\manifests\train_clean.jsonl"
$D1Adapter = Join-Path $RepoRoot "outputs\aic_rgbtir_phase18_d1_probe_returned_20260814\extracted\outputs\aic_rgbtir_phase18_d1_probe_v1\probe_candidates\D1_L050\adapter.pt"
$TokenizerRoot = "C:\Users\12525\.cache\huggingface\hub\models--Qwen--Qwen3-VL-8B-Instruct\snapshots\0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

$Required = @(
    (Join-Path $ManifestRoot "repair_probe_train.jsonl"),
    (Join-Path $ManifestRoot "repair_dev.jsonl"),
    (Join-Path $ManifestRoot "repair_full_train.jsonl"),
    (Join-Path $ManifestRoot "official_val.jsonl"),
    $TrainClean,
    $D1Adapter,
    (Join-Path $TokenizerRoot "tokenizer.json"),
    (Join-Path $TokenizerRoot "tokenizer_config.json")
)
foreach ($Path in $Required) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing bundle asset: $Path" }
}

if (-not (Test-Path -LiteralPath $StageParent -PathType Container)) {
    New-Item -ItemType Directory -Path $StageParent -Force | Out-Null
}
if (Test-Path -LiteralPath $StageRoot) {
    $ResolvedStage = (Resolve-Path -LiteralPath $StageRoot).Path
    $ResolvedParent = (Resolve-Path -LiteralPath $StageParent).Path
    if (-not $ResolvedStage.StartsWith($ResolvedParent, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe staging path: $ResolvedStage"
    }
    Remove-Item -LiteralPath $ResolvedStage -Recurse -Force
}

$Directories = @(
    "src\aic_rgbtir", "tools\cloud", "tests", "configs", "configs\phase18_candidates", "assets\manifests",
    "assets\d1_probe", "assets\tokenizer", "reports"
)
foreach ($Directory in $Directories) {
    New-Item -ItemType Directory -Path (Join-Path $StageRoot $Directory) -Force | Out-Null
}

Get-ChildItem -LiteralPath (Join-Path $RepoRoot "src\aic_rgbtir") -Filter "*.py" -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $StageRoot "src\aic_rgbtir")
}
$Files = @(
    "tools\run_rgbtir_phase16.py",
    "tools\run_rgbtir_phase18_full.py",
    "tools\archive_rgbtir_phase18_stage.py",
    "tools\run_rgbtir_query_interface_smoke.py",
    "tools\run_rgbtir_query_real_smoke.py",
    "tools\download_exact_asset.py",
    "tools\cloud\extract_rgbt_groundbench.sh",
    "tools\cloud\install_rgbtir_phase18_env.sh",
    "tools\cloud\run_rgbtir_phase18_probe.sh",
    "tools\cloud\run_rgbtir_phase18_full_gated.sh",
    "tools\cloud\finalize_rgbtir_phase18_full.sh",
    "configs\aic_rgbtir_phase18.cloud.example.yaml",
    "configs\aic_rgbtir_phase18_full.cloud.example.yaml",
    "configs\aic_rgbtir_phase18_q0.cloud.yaml",
    "configs\aic_rgbtir_phase18_q0_real.cloud.yaml",
    "requirements-rgbtir-phase1.txt",
    "tests\test_rgbtir_phase15.py",
    "tests\test_rgbtir_phase16.py",
    "tests\test_rgbtir_phase18.py",
    "tests\test_rgbtir_phase18_full.py",
    "tests\test_rgbtir_phase18_recovery.py",
    "tests\test_rgbtir_query_interface_smoke.py",
    "tests\test_rgbtir_phase18_bundle_contract.py",
    "reports\AIC_RGB_TIR_PHASE18_Q0_AND_FULL_REVISED_PLAN_2026_08_14.md"
    "reports\aic_rgbtir_phase18r_5090_prerental_readiness_2026_08_15.md"
)
foreach ($Relative in $Files) {
    $Source = Join-Path $RepoRoot $Relative
    $Destination = Join-Path $StageRoot $Relative
    New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Destination
}
Get-ChildItem -LiteralPath (Join-Path $RepoRoot "configs\phase18_candidates") -Filter "*.yaml" -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $StageRoot "configs\phase18_candidates")
}
foreach ($Name in @("repair_probe_train.jsonl", "repair_dev.jsonl", "repair_full_train.jsonl", "official_val.jsonl")) {
    Copy-Item -LiteralPath (Join-Path $ManifestRoot $Name) -Destination (Join-Path $StageRoot "assets\manifests")
}
Copy-Item -LiteralPath $TrainClean -Destination (Join-Path $StageRoot "assets\manifests\train_clean.jsonl")
Copy-Item -LiteralPath $D1Adapter -Destination (Join-Path $StageRoot "assets\d1_probe\adapter.pt")
Get-ChildItem -LiteralPath $TokenizerRoot -File | Where-Object { $_.Name -notlike "model-*.safetensors" } | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $StageRoot "assets\tokenizer")
}

$Expected = @{
    "assets/manifests/repair_dev.jsonl" = "B9DFB8C4E47CAD542326BD25D5B2054EA7CADD64E6AA12BA72E2E7B84CD5D1D7"
    "assets/manifests/repair_full_train.jsonl" = "8E2998C9FC3D20BFB06085F9A0439FB9D353E7D19C5AAFB5902127706FB19874"
    "assets/manifests/official_val.jsonl" = "002FFB665CA8D0CBA545F2A7C4C342ED3B5708274922163B82E66955AF8E5E30"
    "assets/manifests/train_clean.jsonl" = "1AC9B1DCBA1FB240A95E756C4ADC8635DB82A523428B4B520B0ACB3C0D5EFBFB"
    "assets/d1_probe/adapter.pt" = "11509951459607F3E7AF70B9BBD21E09272CA587722EBD6DDFBCB9A955DF0E2C"
}
foreach ($Relative in $Expected.Keys) {
    $Actual = (Get-FileHash -LiteralPath (Join-Path $StageRoot $Relative) -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($Actual -ne $Expected[$Relative]) { throw "Bundle hash mismatch for ${Relative}: $Actual" }
}

$Entries = @{}
Get-ChildItem -LiteralPath $StageRoot -Recurse -File | Sort-Object FullName | ForEach-Object {
    $Relative = $_.FullName.Substring($StageRoot.Length + 1).Replace("\", "/")
    $Entries[$Relative] = @{ bytes = $_.Length; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToUpperInvariant() }
}
$ManifestJson = @{
    schema_version = 1
    phase = "1.8R-Q0+STAGED-FULL"
    scope = "QUERY_INTERFACE_SMOKE_THEN_D1_L050_TRAIN_SELECT_ARCHIVE_THEN_FRESH_PROCESS_SEALED_VAL"
    generated_at = "2026-08-15T00:00:00Z"
    files = $Entries
} | ConvertTo-Json -Depth 8
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText(
    (Join-Path $StageRoot "BUNDLE_MANIFEST.json"),
    $ManifestJson + [Environment]::NewLine,
    $Utf8NoBom
)

if (Test-Path -LiteralPath $OutputZip) { Remove-Item -LiteralPath $OutputZip -Force }
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$Archive = [System.IO.Compression.ZipFile]::Open($OutputZip, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    Get-ChildItem -LiteralPath $StageRoot -Recurse -File | Sort-Object FullName | ForEach-Object {
        $Relative = $_.FullName.Substring($StageRoot.Length + 1).Replace("\", "/")
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $Archive, $_.FullName, "aic_rgbtir_phase18_full_bundle/$Relative",
            [System.IO.Compression.CompressionLevel]::Optimal
        ) | Out-Null
    }
}
finally { $Archive.Dispose() }

@{
    status = "PHASE_18R_5090_CLOUD_BUNDLE_READY"
    zip = $OutputZip
    bytes = (Get-Item -LiteralPath $OutputZip).Length
    sha256 = (Get-FileHash -LiteralPath $OutputZip -Algorithm SHA256).Hash.ToUpperInvariant()
    file_count = $Entries.Count + 1
} | ConvertTo-Json -Depth 4
