param(
    [string]$OutputZip = "D:\12525\Documents\pytorch\aic_rgbtir_phase19_d2_cloud_bundle_20260816.zip"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$OutputParent = Split-Path -Parent $OutputZip
$StageRoot = Join-Path $OutputParent "aic_rgbtir_phase19_d2_bundle"
$PreparedRoot = Join-Path $RepoRoot "outputs\aic_rgbtir_phase19_d2_v1"

$Required = @(
    (Join-Path $PreparedRoot "manifests\d2_probe_train.jsonl"),
    (Join-Path $PreparedRoot "manifests\d2_semantic_dev.jsonl"),
    (Join-Path $PreparedRoot "manifests\d2_multiquery_dev.jsonl"),
    (Join-Path $PreparedRoot "assets\teacher_bank\teacher_bank.pt"),
    (Join-Path $PreparedRoot "assets\d1_selected\adapter.pt"),
    (Join-Path $RepoRoot "configs\aic_rgbtir_phase19_d2.cloud.example.yaml"),
    (Join-Path $RepoRoot "tools\run_rgbtir_phase19_d2.py")
)
foreach ($Path in $Required) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing Phase 1.9-D2 bundle asset: $Path"
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

$Directories = @(
    "src\aic_rgbtir", "tools\cloud", "tests", "configs",
    "assets\manifests", "assets\teacher_bank", "assets\d1_selected", "reports"
)
foreach ($Directory in $Directories) {
    New-Item -ItemType Directory -Path (Join-Path $StageRoot $Directory) -Force | Out-Null
}

Get-ChildItem -LiteralPath (Join-Path $RepoRoot "src\aic_rgbtir") -Filter "*.py" -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $StageRoot "src\aic_rgbtir")
}
$Files = @(
    "tools\run_rgbtir_phase16.py",
    "tools\run_rgbtir_phase19_d2.py",
    "tools\download_exact_asset.py",
    "tools\cloud\install_rgbtir_phase19_d2_env.sh",
    "tools\cloud\run_rgbtir_phase19_d2_probe.sh",
    "configs\aic_rgbtir_phase19_d2.cloud.example.yaml",
    "requirements-rgbtir-phase1.txt",
    "tests\test_rgbtir_phase16.py",
    "tests\test_rgbtir_phase18.py",
    "tests\test_rgbtir_phase19_d2.py",
    "reports\aic_rgbtir_phase19_d2_prerental_plan_2026_08_16.md"
)
foreach ($Relative in $Files) {
    $Source = Join-Path $RepoRoot $Relative
    $Destination = Join-Path $StageRoot $Relative
    New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Destination
}
foreach ($Name in @("d2_probe_train.jsonl", "d2_semantic_dev.jsonl", "d2_multiquery_dev.jsonl")) {
    Copy-Item -LiteralPath (Join-Path $PreparedRoot "manifests\$Name") -Destination (Join-Path $StageRoot "assets\manifests")
}
Copy-Item -LiteralPath (Join-Path $PreparedRoot "assets\teacher_bank\teacher_bank.pt") -Destination (Join-Path $StageRoot "assets\teacher_bank")
Copy-Item -LiteralPath (Join-Path $PreparedRoot "assets\d1_selected\adapter.pt") -Destination (Join-Path $StageRoot "assets\d1_selected")

$Expected = @{
    "assets/manifests/d2_probe_train.jsonl" = "B13FF30E0CD22D1B81CF9939B154BE1B38D28A7C3D0BDF8C45464468965C5D3B"
    "assets/manifests/d2_semantic_dev.jsonl" = "B9DFB8C4E47CAD542326BD25D5B2054EA7CADD64E6AA12BA72E2E7B84CD5D1D7"
    "assets/manifests/d2_multiquery_dev.jsonl" = "A54490B10BF9FC1BFB455C8052F883422D65038603E72CF650749E41D4DF363D"
    "assets/teacher_bank/teacher_bank.pt" = "B43CA908381DD42C4421FB3BF1A895DDD1C9396F40ED8B5BE2B51F2D2E966E08"
    "assets/d1_selected/adapter.pt" = "03D2C3D595495162DFCF2F0044CF06CE9B05AE2B52C1186B9A1434F64A5CED06"
}
foreach ($Relative in $Expected.Keys) {
    $Actual = (Get-FileHash -LiteralPath (Join-Path $StageRoot $Relative) -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($Actual -ne $Expected[$Relative]) {
        throw "Phase 1.9-D2 bundle hash mismatch for ${Relative}: $Actual"
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
    phase = "1.9-D2"
    scope = "D1_L050_PLUS_BASE_TIR_GEOMETRY_RETENTION_PROBE_ONLY"
    generated_at = "2026-08-16T00:00:00Z"
    files = $Entries
} | ConvertTo-Json -Depth 8
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText(
    (Join-Path $StageRoot "BUNDLE_MANIFEST.json"),
    $Manifest + [Environment]::NewLine,
    $Utf8NoBom
)

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
            "aic_rgbtir_phase19_d2_bundle/$Relative",
            [System.IO.Compression.CompressionLevel]::Optimal
        ) | Out-Null
    }
}
finally {
    $Archive.Dispose()
}

@{
    status = "PHASE_19_D2_CLOUD_BUNDLE_READY"
    zip = $OutputZip
    bytes = (Get-Item -LiteralPath $OutputZip).Length
    sha256 = (Get-FileHash -LiteralPath $OutputZip -Algorithm SHA256).Hash.ToUpperInvariant()
    file_count = $Entries.Count + 1
} | ConvertTo-Json -Depth 4
