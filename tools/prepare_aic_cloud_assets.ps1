param(
    [string]$DatasetRoot = "",
    [string]$OutputRoot = "D:\12525\Documents\pytorch\aic_cloud_upload_4090_v1",
    [switch]$SkipArchive
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($DatasetRoot)) {
    $DatasetRoot = [Text.Encoding]::UTF8.GetString(
        [Convert]::FromBase64String(
            "RDpc5Yid6LWb5pWw5o2u6ZuGLeWfuuS6juWkp+aooeWei+eahOWkmuaooeaAgeinhuinieeQhuino+S4juaOqOeQhg=="
        )
    )
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Output = New-Item -ItemType Directory -Force -Path $OutputRoot
$Dataset = Resolve-Path -LiteralPath $DatasetRoot
$Archive = Join-Path $Output.FullName "aic_testset_20260708.tar"
$Manifest = Join-Path $Output.FullName "cloud_upload_manifest.json"

function Get-FileSha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToUpperInvariant()
}

$Queries = Join-Path $Dataset "queries\queries.json"
if (-not (Test-Path -LiteralPath $Queries)) {
    throw "queries.json not found: $Queries"
}
foreach ($relative in @("Images\visible", "Images\infrared", "Images\depth")) {
    $path = Join-Path $Dataset $relative
    if (-not (Test-Path -LiteralPath $path)) {
        throw "required directory not found: $path"
    }
}

if (-not $SkipArchive) {
    if (Test-Path -LiteralPath $Archive) {
        Remove-Item -LiteralPath $Archive
    }
    tar.exe -cf $Archive -C $Dataset Images queries
    if ($LASTEXITCODE -ne 0) {
        throw "tar failed with exit code $LASTEXITCODE"
    }
}

$files = @(
    Get-ChildItem -LiteralPath (Join-Path $Dataset "Images") -Recurse -File
    Get-ChildItem -LiteralPath (Join-Path $Dataset "queries") -Recurse -File
)
$archiveInfo = $null
if (Test-Path -LiteralPath $Archive) {
    $archiveInfo = @{
        path = $Archive
        size_bytes = (Get-Item -LiteralPath $Archive).Length
        sha256 = Get-FileSha256 $Archive
        extract_command = "mkdir -p /workspace/aic_data && tar -xf /workspace/upload/aic_testset_20260708.tar -C /workspace/aic_data"
    }
}

$manifestData = @{
    schema = "aic-cloud-upload-assets-v1"
    created_at = (Get-Date).ToUniversalTime().ToString("o")
    dataset_root = $Dataset.Path
    query_count_expected = 9555
    image_group_count_expected = 2000
    file_count = $files.Count
    dataset_size_bytes = ($files | Measure-Object -Property Length -Sum).Sum
    queries_sha256 = Get-FileSha256 $Queries
    archive = $archiveInfo
    repository = @{
        remote = "https://github.com/finalymadeout7433-gif/aic-multimodal-visual-grounding.git"
        branch = "exp/aic-testset-profile-v1"
    }
    cloud_paths = @{
        project_root = "/workspace/aic-multimodal-visual-grounding"
        dataset_root = "/workspace/aic_data"
        output_root = "/workspace/outputs"
        model_root = "/workspace/models"
    }
}

$manifestData | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Manifest -Encoding UTF8
Write-Host "Prepared cloud assets:"
Write-Host "  manifest: $Manifest"
if ($archiveInfo) {
    Write-Host "  archive:  $Archive"
    Write-Host "  sha256:   $($archiveInfo.sha256)"
}
