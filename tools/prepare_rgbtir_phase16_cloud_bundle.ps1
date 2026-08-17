param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$OutputZip = "D:\12525\Documents\pytorch\aic_rgbtir_phase16_prerental_bundle_20260813.zip"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path -LiteralPath $RepoRoot).Path
$staging = Join-Path $repo "outputs\aic_rgbtir_phase16_cloud_bundle_20260813"
$phase0ManifestRoot = Join-Path $repo "outputs\aic_rgbtir_phase0_v1\manifests"

$required = @(
    (Join-Path $phase0ManifestRoot "train_clean.jsonl"),
    (Join-Path $phase0ManifestRoot "val_official.jsonl"),
    (Join-Path $repo "tools\run_rgbtir_phase16.py"),
    (Join-Path $repo "configs\aic_rgbtir_phase16.cloud.example.yaml")
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required Phase 1.6 bundle input is missing: $path"
    }
}

if (Test-Path -LiteralPath $staging) {
    $resolved = (Resolve-Path -LiteralPath $staging).Path
    if (-not $resolved.StartsWith($repo, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to replace staging outside repository: $resolved"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}

$repoStage = Join-Path $staging "repo"
New-Item -ItemType Directory -Path (Join-Path $repoStage "src") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $repoStage "tools\cloud") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $repoStage "configs") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $repoStage "tests") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $staging "manifests") -Force | Out-Null

Copy-Item -LiteralPath (Join-Path $repo "src\aic_rgbtir") -Destination (Join-Path $repoStage "src\aic_rgbtir") -Recurse
Copy-Item -LiteralPath (Join-Path $repo "tools\run_rgbtir_phase16.py") -Destination (Join-Path $repoStage "tools\run_rgbtir_phase16.py")
Copy-Item -LiteralPath (Join-Path $repo "tools\cloud\install_rgbtir_phase16_env.sh") -Destination (Join-Path $repoStage "tools\cloud\install_rgbtir_phase16_env.sh")
Copy-Item -LiteralPath (Join-Path $repo "tools\cloud\run_rgbtir_phase16_gated.sh") -Destination (Join-Path $repoStage "tools\cloud\run_rgbtir_phase16_gated.sh")
Copy-Item -LiteralPath (Join-Path $repo "configs\aic_rgbtir_phase16.cloud.example.yaml") -Destination (Join-Path $repoStage "configs\aic_rgbtir_phase16.cloud.local.yaml")
Copy-Item -LiteralPath (Join-Path $repo "requirements-rgbtir-phase1.txt") -Destination $repoStage
Copy-Item -Path (Join-Path $repo "tests\test_rgbtir_*.py") -Destination (Join-Path $repoStage "tests")
Copy-Item -LiteralPath (Join-Path $phase0ManifestRoot "train_clean.jsonl") -Destination (Join-Path $staging "manifests\train_clean.jsonl")
Copy-Item -LiteralPath (Join-Path $phase0ManifestRoot "val_official.jsonl") -Destination (Join-Path $staging "manifests\val_official.jsonl")
Copy-Item -LiteralPath (Join-Path $repo "reports\aic_rgbtir_phase16_prerental_readiness_2026_08_13.md") -Destination (Join-Path $staging "PHASE16_READINESS.md")

Get-ChildItem -LiteralPath $staging -Recurse -Directory -Filter "__pycache__" | ForEach-Object {
    $resolvedCache = (Resolve-Path -LiteralPath $_.FullName).Path
    if (-not $resolvedCache.StartsWith($staging, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove cache outside staging: $resolvedCache"
    }
    Remove-Item -LiteralPath $resolvedCache -Recurse -Force
}

$hashRows = Get-ChildItem -LiteralPath $staging -Recurse -File | Sort-Object FullName | ForEach-Object {
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
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$stream = [System.IO.File]::Open($zipPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
try {
    $archive = [System.IO.Compression.ZipArchive]::new($stream, [System.IO.Compression.ZipArchiveMode]::Create, $false)
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
