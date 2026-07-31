param(
    [Parameter(Mandatory = $true)]
    [int]$InitialProcessId,

    [string]$Python = "",

    [string]$Config = "configs\gdino_spatial_ltr_v1.local.yaml"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $ProjectRoot

if ([string]::IsNullOrWhiteSpace($Python)) {
    if (-not [string]::IsNullOrWhiteSpace($env:AIC_BASELINE_PYTHON)) {
        $Python = $env:AIC_BASELINE_PYTHON
    }
    else {
        $Python = (Get-Command python -ErrorAction Stop).Source
    }
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python interpreter does not exist: $Python"
}
$Python = (Resolve-Path -LiteralPath $Python).Path

function Write-Stage {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Output "[$timestamp] $Message"
}

function Wait-GpuMemory {
    param(
        [int]$MinimumMiB = 5120,
        [int]$MaximumMinutes = 30
    )
    $deadline = (Get-Date).AddMinutes($MaximumMinutes)
    while ((Get-Date) -lt $deadline) {
        $raw = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
        if ($LASTEXITCODE -ne 0) {
            throw "nvidia-smi preflight failed"
        }
        $freeMiB = [int](($raw | Select-Object -First 1).Trim())
        Write-Stage "GPU free memory: $freeMiB MiB"
        if ($freeMiB -ge $MinimumMiB) {
            return
        }
        Start-Sleep -Seconds 30
    }
    throw "GPU free memory remained below $MinimumMiB MiB for 30 minutes"
}

function Invoke-RankerPhase {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Phase,
        [switch]$AllowHoldout
    )
    $arguments = @(
        "tools\run_ranker_training_round.py",
        "--config",
        $Config,
        "--phase",
        $Phase
    )
    if ($AllowHoldout) {
        $arguments += "--allow-holdout"
    }
    Write-Stage "Starting phase: $Phase"
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Phase failed: $Phase (exit=$LASTEXITCODE)"
    }
    Write-Stage "Completed phase: $Phase"
}

Write-Stage "Waiting for initial 20k cache process PID=$InitialProcessId"
Wait-Process -Id $InitialProcessId
Write-Stage "Initial 20k cache process completed"

Invoke-RankerPhase -Phase "train-pilot"

Wait-GpuMemory
Invoke-RankerPhase -Phase "cache-train-full"

Invoke-RankerPhase -Phase "train-final"

Wait-GpuMemory
Invoke-RankerPhase -Phase "cache-validation"
Invoke-RankerPhase -Phase "evaluate-validation"

$decisionPath = "outputs\gdino_spatial_ltr_v1\evaluations\validation\frozen_decision.json"
$decision = Get-Content -LiteralPath $decisionPath -Raw | ConvertFrom-Json
if (-not $decision.validation_promotion_pass) {
    throw "Validation promotion failed; holdout remains sealed"
}

Wait-GpuMemory
Invoke-RankerPhase -Phase "cache-holdout" -AllowHoldout
Invoke-RankerPhase -Phase "evaluate-holdout" -AllowHoldout

Wait-GpuMemory
Invoke-RankerPhase -Phase "cache-aic"
Invoke-RankerPhase -Phase "submissions"

Write-Stage "Running complete automated test suite"
$pytestOutput = & $Python -m pytest -q 2>&1
$pytestExit = $LASTEXITCODE
$pytestOutput | ForEach-Object { Write-Output $_ }
if ($pytestExit -ne 0) {
    throw "Automated test suite failed (exit=$pytestExit)"
}
$pytestText = $pytestOutput -join [Environment]::NewLine
if ($pytestText -notmatch '(\d+) passed in ([0-9.]+)s') {
    throw "Could not parse the pytest completion summary"
}
$pytestPassed = [int]$Matches[1]
$pytestSeconds = [double]::Parse(
    $Matches[2],
    [Globalization.CultureInfo]::InvariantCulture
)

Write-Stage "Compiling Python sources"
& $Python -m compileall -q src tools tests
if ($LASTEXITCODE -ne 0) {
    throw "Python compileall failed (exit=$LASTEXITCODE)"
}

$submissionSummaryPath = `
    "outputs\gdino_spatial_ltr_v1\submissions\submission_summary.json"
$submissionSummary = `
    Get-Content -LiteralPath $submissionSummaryPath -Raw | ConvertFrom-Json
$selection = $submissionSummary.selection_summary
$invalidBbox = [int][Math]::Round(
    [double]$selection.records * (1.0 - [double]$selection.legal_bbox_rate)
)
$verification = [ordered]@{
    schema_version = 1
    pytest = [ordered]@{
        status = "passed"
        passed = $pytestPassed
        seconds = $pytestSeconds
    }
    compileall = [ordered]@{ status = "passed" }
    submission_audit = [ordered]@{
        status = if ($invalidBbox -eq 0) { "passed" } else { "failed" }
        records = [int]$selection.records
        invalid_bbox = $invalidBbox
        bbox_differences = [int]$selection.switch_count
    }
    code_review = [ordered]@{ status = "pending" }
}
$verificationPath = `
    "outputs\gdino_spatial_ltr_v1\verification_summary.json"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText(
    (Join-Path $ProjectRoot $verificationPath),
    (($verification | ConvertTo-Json -Depth 8) + [Environment]::NewLine),
    $utf8NoBom
)

Invoke-RankerPhase -Phase "summary"

Write-Stage "Building durable Markdown report"
& $Python "tools\build_ranker_training_report.py" `
    "--output-root" "outputs\gdino_spatial_ltr_v1" `
    "--report" "reports\gdino_spatial_ltr_v1_training_report.md" `
    "--summary" "reports\gdino_spatial_ltr_v1_summary.json"
if ($LASTEXITCODE -ne 0) {
    throw "Report generation failed (exit=$LASTEXITCODE)"
}

Write-Stage "Unattended ranker round completed successfully"
