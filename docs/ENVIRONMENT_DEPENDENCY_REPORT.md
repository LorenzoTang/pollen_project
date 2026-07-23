# Environment Dependency Report

## Current Python Environment

- Python version: `3.13.5`
- Python executable: `C:\Users\22577\Desktop\pollen_project\.venv\Scripts\python.exe`

## Checked Packages

The following packages were checked in the current `.venv` environment:

| Package | Status | Version / Error |
|---|---:|---|
| `torch` | Missing | `ModuleNotFoundError: No module named 'torch'` |
| `torchvision` | Missing | `ModuleNotFoundError: No module named 'torchvision'` |
| `rfdetr` | Missing | `ModuleNotFoundError: No module named 'rfdetr'` |
| `pyyaml` (`yaml`) | Missing | `ModuleNotFoundError: No module named 'yaml'` |
| `pytorch_lightning` | Missing | `ModuleNotFoundError: No module named 'pytorch_lightning'` |
| `lightning` | Missing | `ModuleNotFoundError: No module named 'lightning'` |
| `torchmetrics` | Missing | `ModuleNotFoundError: No module named 'torchmetrics'` |
| `tensorboard` | Missing | `ModuleNotFoundError: No module named 'tensorboard'` |
| `wandb` | Missing | `ModuleNotFoundError: No module named 'wandb'` |

## Installed Packages

No packages from the checked list were installed in the current environment.

## Missing Packages

- torch
- torchvision
- rfdetr
- pyyaml
- pytorch_lightning
- lightning
- torchmetrics
- tensorboard
- wandb

## RF-DETR Minimal Training Dependency Notes

For the current RF-DETR training pipeline, the minimal dependency set should include:

1. **Core training stack**
   - `torch`
   - `torchvision`
   - `rfdetr`

2. **Configuration and training utilities**
   - `pyyaml`
   - `pytorch_lightning` or `lightning` depending on the training script import path
   - `torchmetrics`

3. **Logging / experiment tracking**
   - `tensorboard`
   - `wandb` if Weights & Biases logging is enabled

## Current Implication

The current smoke test cannot start because the environment is missing the packages required by `src/train_rfdetr.py`, beginning with `yaml` / `pyyaml`.

## Next Recommended Step

Before any code changes, install or restore the minimal RF-DETR training environment, then rerun the smoke test.