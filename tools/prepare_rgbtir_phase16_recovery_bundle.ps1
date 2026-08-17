param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$OutputZip = "D:\12525\Documents\pytorch\aic_rgbtir_phase16_recovery_bundle_20260813.zip"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path -LiteralPath $RepoRoot).Path
$staging = Join-Path $repo "outputs\aic_rgbtir_phase16_recovery_bundle_20260813"
$manifestRoot = Join-Path $repo "outputs\aic_rgbtir_phase0_v1\manifests"
$required = @(
    (Join-Path $manifestRoot "train_clean.jsonl"),
    (Join-Path $manifestRoot "val_official.jsonl"),
    (Join-Path $repo "tools\run_rgbtir_phase16.py"),
    (Join-Path $repo "tools\run_rgbtir_phase17a_plus.py"),
    (Join-Path $repo "tools\cloud\run_rgbtir_phase16_recovery.sh"),
    (Join-Path $repo "configs\aic_rgbtir_phase16_recovery.cloud.example.yaml"),
    (Join-Path $repo "configs\aic_rgbtir_phase17a_plus_recovery.cloud.example.yaml"),
    (Join-Path $repo "requirements-rgbtir-phase1.txt")
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Recovery bundle input missing: $path" }
}

if (Test-Path -LiteralPath $staging) {
    $resolved = (Resolve-Path -LiteralPath $staging).Path
    if (-not $resolved.StartsWith($repo, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to replace staging outside repo: $resolved"
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
Copy-Item -LiteralPath (Join-Path $repo "tools\run_rgbtir_phase17a_plus.py") -Destination (Join-Path $repoStage "tools\run_rgbtir_phase17a_plus.py")
Copy-Item -LiteralPath (Join-Path $repo "tools\cloud\install_rgbtir_phase16_env.sh") -Destination (Join-Path $repoStage "tools\cloud\install_rgbtir_phase16_env.sh")
Copy-Item -LiteralPath (Join-Path $repo "tools\cloud\run_rgbtir_phase16_recovery.sh") -Destination (Join-Path $repoStage "tools\cloud\run_rgbtir_phase16_recovery.sh")
$phase16Config = Join-Path $repo "configs\aic_rgbtir_phase16_recovery.cloud.example.yaml"
$phase17Config = Join-Path $repo "configs\aic_rgbtir_phase17a_plus_recovery.cloud.example.yaml"
Copy-Item -LiteralPath $phase16Config -Destination (Join-Path $repoStage "configs\aic_rgbtir_phase16_recovery.cloud.example.yaml")
Copy-Item -LiteralPath $phase16Config -Destination (Join-Path $repoStage "configs\aic_rgbtir_phase16_recovery.cloud.local.yaml")
Copy-Item -LiteralPath $phase17Config -Destination (Join-Path $repoStage "configs\aic_rgbtir_phase17a_plus_recovery.cloud.example.yaml")
Copy-Item -LiteralPath $phase17Config -Destination (Join-Path $repoStage "configs\aic_rgbtir_phase17a_plus_recovery.cloud.local.yaml")
Copy-Item -LiteralPath (Join-Path $repo "requirements-rgbtir-phase1.txt") -Destination $repoStage
Copy-Item -Path (Join-Path $repo "tests\test_rgbtir_*.py") -Destination (Join-Path $repoStage "tests")
Copy-Item -LiteralPath (Join-Path $manifestRoot "train_clean.jsonl") -Destination (Join-Path $staging "manifests\train_clean.jsonl")
Copy-Item -LiteralPath (Join-Path $manifestRoot "val_official.jsonl") -Destination (Join-Path $staging "manifests\val_official.jsonl")
Copy-Item -LiteralPath (Join-Path $repo "reports\aic_rgbtir_phase16_recovery_prerental_readiness_2026_08_13.md") -Destination (Join-Path $staging "RECOVERY_READINESS.md")

Get-ChildItem -LiteralPath $staging -Recurse -Directory -Filter "__pycache__" | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Recurse -Force
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
if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $zipPath -CompressionLevel Optimal
$zipHash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash
Set-Content -LiteralPath ($zipPath + ".sha256") -Value "$zipHash  $([System.IO.Path]::GetFileName($zipPath))" -Encoding ASCII
Write-Output "BUNDLE=$zipPath"
Write-Output "BYTES=$((Get-Item -LiteralPath $zipPath).Length)"
Write-Output "SHA256=$zipHash"
