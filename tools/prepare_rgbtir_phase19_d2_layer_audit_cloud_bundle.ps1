param(
    [string]$OutputZip = "D:\12525\Documents\pytorch\aic_rgbtir_phase19_d2_layer_audit_cloud_bundle_20260817.zip"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$OutputParent = Split-Path -Parent $OutputZip
$StageRoot = Join-Path $OutputParent "aic_rgbtir_phase19_d2_layer_audit_bundle"
$PreparedRoot = Join-Path $RepoRoot "outputs\aic_rgbtir_phase19_d2_v1"
$ReturnedRoot = Join-Path $RepoRoot "outputs\aic_rgbtir_phase19_d2_full_returned_20260817\extracted"

$CheckpointHashes = @{
    "checkpoint_00005848.pt" = "82DDB36AFFD06A7C48ACCF6F91E8DE1C000C7307CB1DD8CCD804A31E1882A4BA"
    "checkpoint_00011696.pt" = "6D30D87D0A513D678B6FF1F15AD85C024167B4425A46B7F098EBE896FC2ABFC6"
    "checkpoint_00017544.pt" = "D878E212E32E5D57AAD01AAE1DB21733B0391FBE07E652A84FD39CA936A00455"
    "checkpoint_00023391.pt" = "07090D4BD4B54E2AA672898ED07770292E5069AACBA8D1864F836B16BA708610"
}
$Required = @(
    (Join-Path $PreparedRoot "manifests\d2_semantic_dev.jsonl"),
    (Join-Path $PreparedRoot "manifests\d2_multiquery_dev.jsonl"),
    (Join-Path $PreparedRoot "assets\teacher_bank\teacher_bank.pt"),
    (Join-Path $RepoRoot "configs\aic_rgbtir_phase19_d2_layer_audit.cloud.example.yaml"),
    (Join-Path $RepoRoot "tools\run_rgbtir_phase19_d2_layer_audit.py")
)
foreach ($Name in $CheckpointHashes.Keys) {
    $Required += Join-Path $ReturnedRoot "checkpoints\$Name"
}
foreach ($Path in $Required) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing D2 layer-audit asset: $Path"
    }
}

if (-not (Test-Path -LiteralPath $OutputParent -PathType Container)) {
    New-Item -ItemType Directory -Path $OutputParent -Force | Out-Null
}
if (Test-Path -LiteralPath $StageRoot) {
    $ResolvedStage = (Resolve-Path -LiteralPath $StageRoot).Path
    $ResolvedParent = (Resolve-Path -LiteralPath $OutputParent).Path
    if (-not $ResolvedStage.StartsWith($ResolvedParent, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe staging path: $ResolvedStage"
    }
    Remove-Item -LiteralPath $ResolvedStage -Recurse -Force
}

foreach ($Directory in @(
    "src\aic_rgbtir", "tools\cloud", "tests", "configs",
    "assets\manifests", "assets\teacher_bank", "assets\checkpoints"
)) {
    New-Item -ItemType Directory -Path (Join-Path $StageRoot $Directory) -Force | Out-Null
}
Get-ChildItem -LiteralPath (Join-Path $RepoRoot "src\aic_rgbtir") -Filter "*.py" -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $StageRoot "src\aic_rgbtir")
}
$Files = @(
    "tools\run_rgbtir_phase16.py",
    "tools\run_rgbtir_phase18_full.py",
    "tools\run_rgbtir_phase19_d2.py",
    "tools\run_rgbtir_phase19_d2_layer_audit.py",
    "tools\cloud\install_rgbtir_phase19_d2_env.sh",
    "tools\cloud\run_rgbtir_phase19_d2_layer_audit_gated.sh",
    "configs\aic_rgbtir_phase19_d2_full.cloud.example.yaml",
    "configs\aic_rgbtir_phase19_d2_layer_audit.cloud.example.yaml",
    "requirements-rgbtir-phase1.txt",
    "tests\test_rgbtir_phase19_d2_layer_audit.py"
)
foreach ($Relative in $Files) {
    $Source = Join-Path $RepoRoot $Relative
    $Destination = Join-Path $StageRoot $Relative
    New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Destination
}
Copy-Item -LiteralPath (Join-Path $PreparedRoot "manifests\d2_semantic_dev.jsonl") -Destination (Join-Path $StageRoot "assets\manifests")
Copy-Item -LiteralPath (Join-Path $PreparedRoot "manifests\d2_multiquery_dev.jsonl") -Destination (Join-Path $StageRoot "assets\manifests")
Copy-Item -LiteralPath (Join-Path $PreparedRoot "assets\teacher_bank\teacher_bank.pt") -Destination (Join-Path $StageRoot "assets\teacher_bank")
foreach ($Name in $CheckpointHashes.Keys) {
    Copy-Item -LiteralPath (Join-Path $ReturnedRoot "checkpoints\$Name") -Destination (Join-Path $StageRoot "assets\checkpoints")
}

$Expected = @{
    "assets/manifests/d2_semantic_dev.jsonl" = "B9DFB8C4E47CAD542326BD25D5B2054EA7CADD64E6AA12BA72E2E7B84CD5D1D7"
    "assets/manifests/d2_multiquery_dev.jsonl" = "A54490B10BF9FC1BFB455C8052F883422D65038603E72CF650749E41D4DF363D"
    "assets/teacher_bank/teacher_bank.pt" = "B43CA908381DD42C4421FB3BF1A895DDD1C9396F40ED8B5BE2B51F2D2E966E08"
}
foreach ($Name in $CheckpointHashes.Keys) {
    $Expected["assets/checkpoints/$Name"] = $CheckpointHashes[$Name]
}
foreach ($Relative in $Expected.Keys) {
    $Actual = (Get-FileHash -LiteralPath (Join-Path $StageRoot $Relative) -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($Actual -ne $Expected[$Relative]) {
        throw "D2 layer-audit bundle hash mismatch for ${Relative}: $Actual"
    }
}

$Entries = @{}
Get-ChildItem -LiteralPath $StageRoot -Recurse -File | Sort-Object FullName | ForEach-Object {
    $Relative = $_.FullName.Substring($StageRoot.Length + 1).Replace("\", "/")
    $Entries[$Relative] = @{
        bytes = $_.Length
        sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToUpperInvariant()
    }
}
$Manifest = @{
    schema_version = 1
    phase = "1.9-D2-Layer-Audit"
    scope = "NO_TRAINING_DUAL_DEV_FOUR_CHECKPOINT_LAYER_AUDIT"
    generated_at = "2026-08-17T00:00:00Z"
    files = $Entries
} | ConvertTo-Json -Depth 8
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText((Join-Path $StageRoot "BUNDLE_MANIFEST.json"), $Manifest + [Environment]::NewLine, $Utf8NoBom)

if (Test-Path -LiteralPath $OutputZip) {
    Remove-Item -LiteralPath $OutputZip -Force
}
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$Archive = [System.IO.Compression.ZipFile]::Open($OutputZip, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    Get-ChildItem -LiteralPath $StageRoot -Recurse -File | Sort-Object FullName | ForEach-Object {
        $Relative = $_.FullName.Substring($StageRoot.Length + 1).Replace("\", "/")
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $Archive,
            $_.FullName,
            "aic_rgbtir_phase19_d2_layer_audit_bundle/$Relative",
            [System.IO.Compression.CompressionLevel]::Optimal
        ) | Out-Null
    }
}
finally {
    $Archive.Dispose()
}

@{
    status = "PHASE_19_D2_LAYER_AUDIT_CLOUD_BUNDLE_READY"
    zip = $OutputZip
    bytes = (Get-Item -LiteralPath $OutputZip).Length
    sha256 = (Get-FileHash -LiteralPath $OutputZip -Algorithm SHA256).Hash.ToUpperInvariant()
    file_count = $Entries.Count + 1
} | ConvertTo-Json -Depth 4
