$ErrorActionPreference = "Stop"

$pythonCheck = @'
import torch
import torch.nn.functional as F
import ape
import ape._C
import detectron2
import detrex
import clip

print("APE import: OK")
print("APE CUDA extension: OK")
print("Detectron2:", detectron2.__version__)
print("Detrex:", detrex.__file__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA test result:", torch.arange(1, 9, device="cuda").sum().item())
x = torch.randn(1, 3, 64, 64, device="cuda")
w = torch.randn(8, 3, 3, 3, device="cuda")
y = F.conv2d(x, w, padding=1)
torch.cuda.synchronize()
print("cuDNN:", torch.backends.cudnn.version())
print("cuDNN convolution:", tuple(y.shape))
'@

$pythonCheck | wsl.exe -d Ubuntu-22.04 -u aicuser -- bash -lc `
    "source /home/aicuser/miniforge3/bin/activate /home/aicuser/miniforge3/envs/aic-ape && python -"

wsl.exe -d Ubuntu-22.04 -u aicuser -- bash -lc `
    "source /home/aicuser/miniforge3/bin/activate /home/aicuser/miniforge3/envs/aic-ape && python -m pip check"

