param(
    [string]$ProjectRoot = "D:\12525\Documents\pytorch\baseline_v0",
    [string]$DatasetRoot = ""
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($DatasetRoot)) {
    # Keep the script itself ASCII-compatible for Windows PowerShell 5.1,
    # which otherwise decodes a BOM-less UTF-8 path using the legacy code page.
    $DatasetRoot = [Text.Encoding]::UTF8.GetString(
        [Convert]::FromBase64String(
            "RDpc5Yid6LWb5pWw5o2u6ZuGLeWfuuS6juWkp+aooeWei+eahOWkmuaooeaAgeinhuinieeQhuino+S4juaOqOeQhg=="
        )
    )
}
$Python = "D:\Anaconda\envs\aic-baseline\python.exe"
$Queries = Join-Path $DatasetRoot "queries\queries.json"
$OutputRoot = Join-Path $ProjectRoot "outputs\aic_zero_shot_full_v1"
$Expectations = Join-Path $ProjectRoot "configs\aic_three_model_full_fingerprints.json"
$Drive = $DatasetRoot.Substring(0, 1).ToLowerInvariant()
$WslDatasetRoot = "/mnt/$Drive/" + $DatasetRoot.Substring(3).Replace("\", "/")
$WslQueries = "$WslDatasetRoot/queries/queries.json"

function Test-CompletedRun([string]$RunDirectory, [string]$RunKey) {
    if (-not (Test-Path -LiteralPath $Expectations)) {
        return $false
    }
    & $Python (Join-Path $ProjectRoot "tools\verify_aic_completed_run.py") `
        --run-directory $RunDirectory `
        --expectations $Expectations `
        --run-key $RunKey | Out-Null
    return $LASTEXITCODE -eq 0
}

$MmOutput = Join-Path $OutputRoot "mm_grounding_dino_t"
if (-not (Test-CompletedRun $MmOutput "mm_grounding_dino_t")) {
    & $Python (Join-Path $ProjectRoot "tools\run_aic_hf_full_detection.py") `
        --dataset-root $DatasetRoot `
        --queries $Queries `
        --model-path "D:\AI_Models\huggingface\openmmlab-community\mm-grounding-dino-tiny-o365v1-goldg" `
        --model-name "openmmlab-community/mm-grounding-dino-tiny-o365v1-goldg" `
        --output-dir $MmOutput `
        --dtype float32 `
        --autocast-dtype none `
        --box-threshold 0 `
        --text-threshold 0 `
        --max-candidates 20 `
        --batch-size 1 `
        --resume `
        --finalize
    if ($LASTEXITCODE -ne 0) { throw "MM-Grounding-DINO-T failed" }
    if (-not (Test-CompletedRun $MmOutput "mm_grounding_dino_t")) {
        throw "MM-Grounding-DINO-T output failed completed-run audit"
    }
}

$LlmOutput = Join-Path $OutputRoot "llmdet_swin_t"
if (-not (Test-CompletedRun $LlmOutput "llmdet_swin_t")) {
    & $Python (Join-Path $ProjectRoot "tools\run_aic_hf_full_detection.py") `
        --dataset-root $DatasetRoot `
        --queries $Queries `
        --model-path "D:\AI_Models\derived\fushh7\llmdet-swin-tiny-hf-complete-safetensors" `
        --model-name "fushh7/llmdet-swin-tiny-hf" `
        --custom-modeling-path "D:\AI_Models\repos\LLMDet\hf_model" `
        --output-dir $LlmOutput `
        --dtype float32 `
        --autocast-dtype none `
        --box-threshold 0 `
        --text-threshold 0 `
        --max-candidates 20 `
        --batch-size 1 `
        --resume `
        --finalize
    if ($LASTEXITCODE -ne 0) { throw "LLMDet-Swin-T failed" }
    if (-not (Test-CompletedRun $LlmOutput "llmdet_swin_t")) {
        throw "LLMDet-Swin-T output failed completed-run audit"
    }
}

$ApeOutput = Join-Path $OutputRoot "ape_ti"
if (-not (Test-CompletedRun $ApeOutput "ape_ti")) {
    $ApeCommand = @"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
source /home/aicuser/miniforge3/etc/profile.d/conda.sh
conda activate /home/aicuser/miniforge3/envs/aic-ape
cd /home/aicuser/aic/APE
python /mnt/d/12525/Documents/pytorch/baseline_v0/tools/wsl/run_ape_ti_full_detection.py \
  --project-root /mnt/d/12525/Documents/pytorch/baseline_v0 \
  --ape-root /home/aicuser/aic/APE \
  --config /home/aicuser/aic/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitt_eva02_vlf_lsj1024_cp_16x4_1080k.py \
  --checkpoint /mnt/d/AI_Models/huggingface/shenyunhang/APE-Ti/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitt_eva02_vlf_lsj1024_cp_16x4_1080k_mdl_20240203_230000/model_final.pth \
  --checkpoint-sha256 B5D793E960515A6D1AA4B8A55B61DA990B0B4184B510D3D7AD3BB37526FB8007 \
  --dataset-root '$WslDatasetRoot' \
  --queries '$WslQueries' \
  --output-dir /mnt/d/12525/Documents/pytorch/baseline_v0/outputs/aic_zero_shot_full_v1/ape_ti \
  --score-threshold 0 \
  --max-candidates 20 \
  --batch-size 1 \
  --resume \
  --finalize
"@
    wsl.exe -d Ubuntu-22.04 -u aicuser -- bash -lc $ApeCommand
    if ($LASTEXITCODE -ne 0) { throw "APE-Ti failed" }
    if (-not (Test-CompletedRun $ApeOutput "ape_ti")) {
        throw "APE-Ti output failed completed-run audit"
    }
}

& $Python (Join-Path $ProjectRoot "tools\build_three_model_platform_release.py") `
    --queries $Queries `
    --ape-zip (Join-Path $ApeOutput "submission\predictions_submission.zip") `
    --mm-zip (Join-Path $MmOutput "submission\predictions_submission.zip") `
    --llmdet-zip (Join-Path $LlmOutput "submission\predictions_submission.zip") `
    --output-dir (Join-Path $OutputRoot "platform_upload_ready")
if ($LASTEXITCODE -ne 0) { throw "Platform release packaging failed" }

& $Python (Join-Path $ProjectRoot "tools\analyze_aic_three_model_outputs.py") `
    --ape (Join-Path $ApeOutput "predictions.jsonl") `
    --mm (Join-Path $MmOutput "predictions.jsonl") `
    --llmdet (Join-Path $LlmOutput "predictions.jsonl") `
    --output-json (Join-Path $OutputRoot "three_model_behavior_summary.json") `
    --output-md (Join-Path $ProjectRoot "reports\aic_three_model_zero_shot_behavior.md")
if ($LASTEXITCODE -ne 0) { throw "Three-model behavior analysis failed" }
