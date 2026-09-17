# MMCV2

MMCV2 is a PyTorch-oriented computer-vision foundation library. It provides
image and storage utilities, configurable CNN components, data structures,
evaluation, and training runners.

## Install

```bash
pip install -e ".[dev]"
```

## Development checks

```bash
ruff check .
ruff format --check .
pyright --project pyproject.toml
pytest --cov=mmcv2 --cov-report=term-missing
```

The coverage gate starts at 50% and should only be raised as the suite grows.

## Torchvision classification demo

`examples/torchvision_classification.py` is a complete MMCV2 Runner example
with CIFAR-10/100, MNIST, Fashion-MNIST, and torchvision `FakeData`. Its model
implements `training_step`, `validation_step`, and `test_step`; `EpochBasedRunner` drives
training, while optimizer, AMP, LR scheduling, validation, logging, and
checkpointing are composed from MMCV2 hooks. It supports single-device
training, single-node `MMDistributedDataParallel`, gradient accumulation and
clipping, runner resume, best-checkpoint selection, and evaluation-only runs.

```bash
# Single GPU (downloads CIFAR-10 into ./data)
python examples/torchvision_classification.py \
  --dataset cifar10 --model resnet18 --epochs 20 --amp

# Two GPUs. --batch-size is the per-GPU batch size.
torchrun --standalone --nproc-per-node=2 \
  examples/torchvision_classification.py \
  --dataset cifar10 --model resnet18 --epochs 20 --amp

# Resume, or evaluate a saved checkpoint
python examples/torchvision_classification.py \
  --dataset cifar10 --model resnet18 --epochs 30 \
  --resume work_dir/classification/latest.pth --amp
python examples/torchvision_classification.py \
  --dataset cifar10 --model resnet18 \
  --resume work_dir/classification/best_classification_accuracy_epoch_20.pth \
  --evaluate

# Fast offline CPU smoke test
python examples/torchvision_classification.py \
  --dataset fake --device cpu --epochs 1 --workers 0 \
  --fake-train-size 128 --fake-val-size 64
```

Checkpoints and JSON/text runner logs are written to
`work_dir/classification` by default. Best checkpoint filenames include the
metric and epoch. The effective batch size is
`batch-size * number-of-processes * accumulation-steps`.
