"""Train a torchvision classifier with MMCV2's runner and hook system.

Single GPU::

    python examples/torchvision_classification.py --dataset cifar10

Single-node multi-GPU (one process per GPU)::

    torchrun --standalone --nproc-per-node=2 \
        examples/torchvision_classification.py --dataset cifar10

Use ``--dataset fake`` for a quick, download-free smoke test.
"""

from __future__ import annotations

import argparse
import os
import random
from collections.abc import Callable, Iterator, Sequence, Sized
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from mmcv2.distributed import get_rank, get_world_size, init_dist, is_main_process
from mmcv2.evaluator import BaseMetric, Evaluator
from mmcv2.parallel import MMDistributedDataParallel
from mmcv2.runner import (
    AmpOptimizerHook,
    CheckpointHook,
    DistEvalHook,
    DistSamplerSeedHook,
    EpochBasedRunner,
    EvalHook,
    OptimizerHook,
)
from mmcv2.utils import get_logger
from torch import distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torchvision import datasets, models, transforms


DATASETS = {
    "cifar10": (datasets.CIFAR10, 10, False),
    "cifar100": (datasets.CIFAR100, 100, False),
    "mnist": (datasets.MNIST, 10, True),
    "fashion_mnist": (datasets.FashionMNIST, 10, True),
}


class TinyClassifier(nn.Module):
    """Small backbone that makes the demo practical on CPU and small images."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).flatten(1))


class ClassificationModel(nn.Module):
    """Adapt a plain torchvision model to MMCV2's step-based model API."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)

    def _prepare_batch(self, data_batch: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        images, targets = data_batch
        device = next(self.parameters()).device
        return images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

    def _step(self, data_batch: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images, targets = self._prepare_batch(data_batch)
        logits = self(images)
        return F.cross_entropy(logits, targets), logits, targets

    def training_step(self, data_batch: Sequence[torch.Tensor], batch_idx: int) -> dict[str, Any]:
        del batch_idx  # Optimization is owned by MMCV2's OptimizerHook.
        loss, logits, targets = self._step(data_batch)
        accuracy = (logits.argmax(dim=1) == targets).float().mean() * 100
        return {
            "loss": loss,
            "log_vars": {"loss": loss.detach(), "accuracy": accuracy.detach()},
            "num_samples": targets.size(0),
        }

    def validation_step(self, data_batch: Sequence[torch.Tensor], batch_idx: int) -> list[dict[str, float | int]]:
        del batch_idx
        images, targets = self._prepare_batch(data_batch)
        logits = self(images)
        losses = F.cross_entropy(logits, targets, reduction="none")
        predictions = logits.argmax(dim=1)
        return [
            {"pred": int(prediction), "target": int(target), "loss": float(loss)}
            for prediction, target, loss in zip(predictions, targets, losses)
        ]

    def test_step(self, data_batch: Sequence[torch.Tensor]) -> list[dict[str, float | int]]:
        return self.validation_step(data_batch, batch_idx=0)


class ClassificationMetric(BaseMetric):
    """Compute globally aggregated loss and top-1 accuracy."""

    default_prefix = "classification"

    def process(self, data_batch: Any, data_samples: Sequence[Any]) -> None:
        del data_batch
        self.results.extend((int(sample["pred"] == sample["target"]), float(sample["loss"])) for sample in data_samples)

    def compute_metrics(self, results: list[Any]) -> dict[str, float]:
        if not results:
            return {"accuracy": 0.0, "loss": 0.0}
        correct = sum(result[0] for result in results)
        loss = sum(result[1] for result in results)
        return {"accuracy": 100.0 * correct / len(results), "loss": loss / len(results)}


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation data without DistributedSampler's padded duplicates."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.rank = get_rank()
        self.world_size = get_world_size()

    def __iter__(self) -> Iterator[int]:
        dataset = cast(Sized, self.dataset)
        return iter(range(self.rank, len(dataset), self.world_size))

    def __len__(self) -> int:
        dataset = cast(Sized, self.dataset)
        return (len(dataset) - self.rank + self.world_size - 1) // self.world_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=[*DATASETS, "fake"], default="cifar10")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model", choices=["tiny_cnn", "resnet18", "mobilenet_v3_small"], default="tiny_cnn")
    parser.add_argument("--pretrained", action="store_true", help="Start a torchvision model from ImageNet weights.")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size per process/GPU.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--clip-grad-norm", type=float, default=0.0, help="Zero disables gradient clipping.")
    parser.add_argument("--amp", action="store_true", help="Use MMCV2's native CUDA AMP optimizer hook.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("work_dir/classification"))
    parser.add_argument("--resume", type=Path, help="Resume runner, optimizer, AMP, RNG, and best-metric state.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate --resume without training.")
    parser.add_argument("--print-freq", type=int, default=50, help="Set to zero to disable iteration logging.")
    parser.add_argument("--max-keep-checkpoints", type=int, default=3)
    parser.add_argument("--fake-train-size", type=int, default=1024)
    parser.add_argument("--fake-val-size", type=int, default=256)
    args = parser.parse_args()
    if args.pretrained and args.model == "tiny_cnn":
        parser.error("--pretrained requires --model resnet18 or mobilenet_v3_small")
    if args.evaluate and args.resume is None:
        parser.error("--evaluate requires --resume")
    for name in (
        "image_size",
        "epochs",
        "batch_size",
        "accumulation_steps",
        "max_keep_checkpoints",
        "fake_train_size",
        "fake_val_size",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def setup_device(requested: str) -> tuple[torch.device, bool]:
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        if requested == "cpu":
            init_dist("pytorch", backend="gloo")
            return torch.device("cpu"), True
        if not torch.cuda.is_available():
            raise RuntimeError(
                "torchrun GPU training requested, but CUDA is unavailable (use --device cpu for a CPU test)"
            )
        init_dist("pytorch")
        return torch.device("cuda", int(os.environ["LOCAL_RANK"])), True

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    use_cuda = torch.cuda.is_available() and requested != "cpu"
    return torch.device("cuda", 0) if use_cuda else torch.device("cpu"), False


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def build_transforms(
    dataset_name: str, image_size: int, pretrained: bool
) -> tuple[transforms.Compose, transforms.Compose]:
    grayscale = dataset_name in {"mnist", "fashion_mnist"}
    prefix: list[Callable[[Any], Any]] = [transforms.Grayscale(num_output_channels=3)] if grayscale else []
    mean = (0.485, 0.456, 0.406) if pretrained else (0.5, 0.5, 0.5)
    std = (0.229, 0.224, 0.225) if pretrained else (0.5, 0.5, 0.5)
    train_ops: list[Callable[[Any], Any]] = [
        *prefix,
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
    ]
    if not grayscale:
        train_ops.append(transforms.RandomHorizontalFlip())
    train_ops.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    val_ops = [
        *prefix,
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(train_ops), transforms.Compose(val_ops)


def build_datasets(args: argparse.Namespace, distributed: bool) -> tuple[Dataset, Dataset, int]:
    train_transform, val_transform = build_transforms(args.dataset, args.image_size, args.pretrained)
    if args.dataset == "fake":
        train_set = datasets.FakeData(
            size=args.fake_train_size,
            image_size=(3, args.image_size, args.image_size),
            num_classes=10,
            transform=train_transform,
            random_offset=0,
        )
        val_set = datasets.FakeData(
            size=args.fake_val_size,
            image_size=(3, args.image_size, args.image_size),
            num_classes=10,
            transform=val_transform,
            random_offset=args.fake_train_size,
        )
        return train_set, val_set, 10

    dataset_cls, num_classes, _ = DATASETS[args.dataset]
    if not distributed or is_main_process():
        train_set = dataset_cls(root=args.data_root, train=True, download=True, transform=train_transform)
        val_set = dataset_cls(root=args.data_root, train=False, download=True, transform=val_transform)
    if distributed:
        dist.barrier()
        if not is_main_process():
            train_set = dataset_cls(root=args.data_root, train=True, download=False, transform=train_transform)
            val_set = dataset_cls(root=args.data_root, train=False, download=False, transform=val_transform)
    return train_set, val_set, num_classes


def build_backbone(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if name == "tiny_cnn":
        return TinyClassifier(num_classes)
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        last = model.classifier[-1]
        if not isinstance(last, nn.Linear):
            raise TypeError("Expected MobileNetV3's last classifier layer to be Linear")
        model.classifier[-1] = nn.Linear(last.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model: {name}")


def build_loaders(
    args: argparse.Namespace, train_set: Dataset, val_set: Dataset, device: torch.device, distributed: bool
) -> tuple[DataLoader, DataLoader]:
    train_sampler = DistributedSampler(train_set, shuffle=True, seed=args.seed) if distributed else None
    val_sampler = DistributedEvalSampler(val_set) if distributed else None
    common = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=val_sampler, shuffle=False, **common)
    return train_loader, val_loader


def build_optimizer_hook(args: argparse.Namespace, amp_enabled: bool) -> OptimizerHook:
    options: dict[str, Any] = {"cumulative_iters": args.accumulation_steps}
    if args.clip_grad_norm > 0:
        options["grad_clip"] = {"max_norm": args.clip_grad_norm, "norm_type": 2}
    if amp_enabled:
        return AmpOptimizerHook(dtype="float16", loss_scale="dynamic", **options)
    return OptimizerHook(**options)


def main() -> None:
    args = parse_args()
    device, distributed = setup_device(args.device)
    try:
        seed_everything(args.seed + get_rank(), args.deterministic)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logger = get_logger("mmcv2.classification", log_file=str(args.output_dir / "train.log"))
        logger.info(
            "device=%s world_size=%d dataset=%s model=%s effective_batch=%d",
            device,
            get_world_size(),
            args.dataset,
            args.model,
            args.batch_size * get_world_size() * args.accumulation_steps,
        )

        train_set, val_set, num_classes = build_datasets(args, distributed)
        train_loader, val_loader = build_loaders(args, train_set, val_set, device, distributed)

        backbone = None
        if not distributed or not args.pretrained or is_main_process():
            backbone = build_backbone(args.model, num_classes, args.pretrained)
        if distributed and args.pretrained:
            dist.barrier()
            if backbone is None:
                backbone = build_backbone(args.model, num_classes, args.pretrained)
        assert backbone is not None
        model: nn.Module = ClassificationModel(backbone).to(device)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
        if distributed:
            model = MMDistributedDataParallel(
                model,
                device_ids=[cast(int, device.index)] if device.type == "cuda" else None,
            )

        runner = EpochBasedRunner(
            model,
            optimizer=optimizer,
            work_dir=str(args.output_dir),
            logger=logger,
            meta={
                "seed": args.seed,
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            },
            max_epochs=args.epochs,
        )
        amp_enabled = args.amp and device.type == "cuda"
        if args.amp and not amp_enabled:
            logger.warning("--amp is only enabled on CUDA; using the standard OptimizerHook")
        runner.register_training_hooks(
            lr_config={"policy": "CosineAnnealing", "min_lr": 0.0},
            optimizer_config=build_optimizer_hook(args, amp_enabled),
            checkpoint_config=CheckpointHook(
                interval=1,
                max_keep_ckpts=args.max_keep_checkpoints,
                save_last=True,
            ),
            log_config=(
                {"interval": args.print_freq, "hooks": [{"type": "TextLoggerHook"}]} if args.print_freq > 0 else None
            ),
        )
        if distributed:
            runner.register_hook(DistSamplerSeedHook())

        evaluator = Evaluator(ClassificationMetric())
        eval_hook: EvalHook = (
            DistEvalHook(val_loader, evaluator, interval=1, save_best="classification/accuracy", rule="greater")
            if distributed
            else EvalHook(val_loader, evaluator, interval=1, save_best="classification/accuracy", rule="greater")
        )
        # Evaluate before CheckpointHook so latest.pth contains the current best-metric metadata.
        runner.register_hook(eval_hook, priority="ABOVE_NORMAL")

        if args.resume:
            runner.resume(str(args.resume))
        if args.evaluate:
            runner.call_hook("before_run")
            eval_hook._do_evaluate(runner)
            runner.call_hook("after_run")
            metrics = runner.log_buffer.output
            logger.info(
                "validation: loss=%.4f accuracy=%.2f%%",
                metrics["classification/loss"],
                metrics["classification/accuracy"],
            )
            return

        runner.run([train_loader], [("train", 1)])
        best = cast(dict[str, Any], runner.meta).get("hook_msgs", {}).get("best_score")
        if best is not None:
            logger.info("training complete; best validation accuracy: %.2f%%", best)
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
