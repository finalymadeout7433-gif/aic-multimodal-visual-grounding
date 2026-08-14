$ErrorActionPreference = "Stop"

$distro = "Ubuntu-22.04"
$linuxUser = "aicuser"
$condaRoot = "/home/aicuser/miniforge3"
$envName = "aic-ape"
$apeRoot = "/home/aicuser/aic/APE"

$linuxCommand = @"
set -e
source '$condaRoot/etc/profile.d/conda.sh'
conda activate '$envName'
cd '$apeRoot'
echo
echo 'AIC APE environment is ready.'
echo "Python: `$(python --version 2>&1)"
echo "Workspace: `$(pwd)"
echo "GPU: `$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
echo
exec bash -i
"@

Write-Host "Starting $distro / $envName ..."
wsl.exe -d $distro -u $linuxUser -- bash -lc $linuxCommand

