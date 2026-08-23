param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$OutputZip = "D:\12525\Documents\pytorch\aic_rgbtir_phase1_prerental_bundle_20260812.zip"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path -LiteralPath $RepoRoot).Path
$staging = Join-Path $repo "outputs\aic_rgbtir_phase1_cloud_bundle_20260812"
$manifestRoot = Join-Path $repo "outputs\aic_rgbtir_phase0_v1\manifests"
$phase0Summary = Join-Path $repo "outputs\aic_rgbtir_phase0_v1\run_summary.json"

if (-not (Test-Path -LiteralPath $phase0Summary)) {
    throw "Phase 0 summary is missing: $phase0Summary"
}
$summary = Get-Content -LiteralPath $phase0Summary -Raw -Encoding UTF8 | ConvertFrom-Json
if ($summary.status -ne "PHASE_0_GO") {
    throw "Phase 0 is not GO: $($summary.status)"
}

if (Test-Path -LiteralPath $staging) {
    $resolved = (Resolve-Path -LiteralPath $staging).Path
    if (-not $resolved.StartsWith($repo, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to replace staging outside repository: $resolved"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}

$repoStage = Join-Path $staging "repo"
New-Item -ItemType Directory -Path $repoStage -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $staging "manifests") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $repoStage "src") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $repoStage "tools\cloud") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $repoStage "configs") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $repoStage "tests") -Force | Out-Null

Copy-Item -LiteralPath (Join-Path $repo "src\aic_rgbtir") -Destination (Join-Path $repoStage "src\aic_rgbtir") -Recurse
Copy-Item -LiteralPath (Join-Path $repo "tools\run_rgbtir_phase1.py") -Destination (Join-Path $repoStage "tools\run_rgbtir_phase1.py") -Force
Copy-Item -LiteralPath (Join-Path $repo "tools\cloud\install_rgbtir_phase1_env.sh") -Destination (Join-Path $repoStage "tools\cloud\install_rgbtir_phase1_env.sh") -Force
Copy-Item -LiteralPath (Join-Path $repo "tools\cloud\run_rgbtir_phase1_gated.sh") -Destination (Join-Path $repoStage "tools\cloud\run_rgbtir_phase1_gated.sh") -Force
Copy-Item -LiteralPath (Join-Path $repo "tools\cloud\extract_rgbt_groundbench.sh") -Destination (Join-Path $repoStage "tools\cloud\extract_rgbt_groundbench.sh") -Force
Copy-Item -LiteralPath (Join-Path $repo "configs\aic_rgbtir_phase1.cloud.example.yaml") -Destination (Join-Path $repoStage "configs\aic_rgbtir_phase1.cloud.local.yaml") -Force
Copy-Item -LiteralPath (Join-Path $repo "requirements-rgbtir-phase1.txt") -Destination $repoStage -Force
Copy-Item -LiteralPath (Join-Path $repo "tests\test_rgbtir_phase1.py") -Destination (Join-Path $repoStage "tests\test_rgbtir_phase1.py") -Force
Copy-Item -LiteralPath (Join-Path $repo "reports\aic_rgbtir_phase1_prerental_readiness_2026_08_12.md") -Destination (Join-Path $staging "PHASE1_READINESS.md") -Force
Copy-Item -LiteralPath (Join-Path $repo "reports\aic_rgbtir_phase1_cloud_runbook_2026_08_12.md") -Destination (Join-Path $staging "CLOUD_RUNBOOK.md") -Force
Copy-Item -LiteralPath $phase0Summary -Destination (Join-Path $staging "phase0_run_summary.json") -Force
Copy-Item -Path (Join-Path $manifestRoot "*.jsonl") -Destination (Join-Path $staging "manifests") -Force
Copy-Item -Path (Join-Path $manifestRoot "*.json") -Destination (Join-Path $staging "manifests") -Force

Get-ChildItem -LiteralPath $staging -Recurse -Directory -Filter "__pycache__" | ForEach-Object {
    $resolvedCache = (Resolve-Path -LiteralPath $_.FullName).Path
    if (-not $resolvedCache.StartsWith($staging, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove cache outside staging: $resolvedCache"
    }
    Remove-Item -LiteralPath $resolvedCache -Recurse -Force
}

$hashRows = Get-ChildItem -LiteralPath $staging -Recurse -File | ForEach-Object {
    [PSCustomObject]@{
        path = $_.FullName.Substring($staging.Length + 1).Replace("\", "/")
        bytes = $_.Length
        sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
    }
}
$hashRows | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $staging "SHA256_MANIFEST.json") -Encoding UTF8

$zipPath = [System.IO.Path]::GetFullPath($OutputZip)
if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}

# Compress-Archive stores Windows path separators in entry names on some
# PowerShell/.NET combinations.  Linux unzip then treats those entries as
# malformed.  Build the archive explicitly with POSIX entry names because this
# bundle is consumed by the Featurize Linux instance.
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$stream = [System.IO.File]::Open(
    $zipPath,
    [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::ReadWrite,
    [System.IO.FileShare]::None
)
try {
    $archive = [System.IO.Compression.ZipArchive]::new(
        $stream,
        [System.IO.Compression.ZipArchiveMode]::Create,
        $false
    )
    try {
        Get-ChildItem -LiteralPath $staging -Recurse -File | Sort-Object FullName | ForEach-Object {
            $entryName = $_.FullName.Substring($staging.Length + 1).Replace("\", "/")
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
}
finally {
    $stream.Dispose()
}
$zipHash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash
Set-Content -LiteralPath ($zipPath + ".sha256") -Value "$zipHash  $([System.IO.Path]::GetFileName($zipPath))" -Encoding ASCII
Write-Output "BUNDLE=$zipPath"
Write-Output "BYTES=$((Get-Item -LiteralPath $zipPath).Length)"
Write-Output "SHA256=$zipHash"
