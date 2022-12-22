from __future__ import annotations

import argparse
import os
from pprint import pprint
from typing import Any

import timm  # type: ignore
import torch
import torch.nn as nn
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
from torchvision.datasets import ImageFolder  # type: ignore

from options import get_args
from utils.utils import ImageTransform

# Settings
OPT = "adam"  # adam, sgd
WEIGHT_DECAY = 0.00001
MOMENTUM = 0.9  # only when OPT is sgd
BASE_LR = 0.0001
LR_SCHEDULER = "reduce_on_plateau"  # step, multistep, reduce_on_plateau
LR_DECAY_RATE = 0.1
LR_STEP_SIZE = 5  # only when LR_SCHEDULER is step
LR_STEP_MILESTONES = [10, 15]  # only when LR_SCHEDULER is multistep

# get arguments
args = get_args()


class CreateDataloaders(LightningDataModule):
    """Create dataloader"""

    def __init__(
        self,
        root_dir: str,
        img_size: tuple[int, int] = (224, 224),
        batch_size: int = 16,
        num_workers: int = 10,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = ImageFolder(
            root=os.path.join(root_dir, "train"),
            transform=ImageTransform(is_train=True, img_size=self.img_size),
        )
        self.val_dataset = ImageFolder(
            root=os.path.join(root_dir, "test"),
            transform=ImageTransform(is_train=False, img_size=self.img_size),
        )
        self.classes = self.train_dataset.classes
        self.class_to_idx = self.train_dataset.class_to_idx

    def train_dataloader(self) -> DataLoader:  # type: ignore
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:  # type: ignore
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        return dataloader


# pylint: disable=arguments-differ
class AnomalyClassifier(LightningModule):
    """Defines the model and the training steps"""

    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = False,
        num_classes: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()  # type: ignore

        # model definition
        self.model = timm.create_model(
            model_name=model_name, pretrained=pretrained, num_classes=num_classes
        )
        self.criterion = nn.CrossEntropyLoss()

        # metircs initialization
        self.train_acc = Accuracy()
        self.train_f1 = F1Score(num_classes=num_classes, average="weighted")

        self.val_acc = Accuracy()
        self.val_f1 = F1Score(num_classes=num_classes, average="weighted")

    def forward(self, x: torch.Tensor) -> Any:
        return self.model(x)

    def training_step(self, batch: torch.Tensor, _):  # type: ignore
        x, target = batch
        out = self(x)
        _, pred = out.max(1)

        loss = self.criterion(out, target)
        acc = self.train_acc(pred, target)
        f1 = self.train_f1(pred, target)

        self.log_dict(
            {
                "train/loss": loss,
                "train/acc": acc,
                "train/f1": f1,
            },
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        return loss

    def validation_step(self, batch, _):  # type: ignore
        x, target = batch
        out = self(x)
        loss = self.criterion(out, target)

        _, pred = out.max(1)
        acc = self.val_acc(pred, target)
        f1 = self.val_f1(pred, target)

        self.log_dict(
            {
                "val/loss": loss,
                "val/acc": acc,
                "val/f1": f1,
            },
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

        return loss

    def configure_optimizers(self):  # type: ignore
        optimizer = get_optimizer(self.parameters())
        lr_scheduler_config = get_lr_scheduler_config(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


def get_optimizer(parameters) -> torch.optim.Optimizer:  # type: ignore
    if OPT == "adam":
        optimizer = torch.optim.Adam(parameters, lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    elif OPT == "sgd":
        optimizer = torch.optim.SGD(
            parameters, lr=BASE_LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM
        )  # type: ignore
    else:
        raise NotImplementedError()

    return optimizer


def get_lr_scheduler_config(optimizer: torch.optim.Optimizer) -> dict:  # type: ignore
    if LR_SCHEDULER == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=LR_STEP_SIZE, gamma=LR_DECAY_RATE
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
    elif LR_SCHEDULER == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=LR_STEP_MILESTONES, gamma=LR_DECAY_RATE
        )  # type: ignore
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
    elif LR_SCHEDULER == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=10, threshold=0.0001
        )  # type: ignore
        lr_scheduler_config = {
            "scheduler": scheduler,
            "monitor": "train/loss",
            "interval": "epoch",
            "frequency": 1,
        }
    else:
        raise NotImplementedError

    return lr_scheduler_config


def get_basic_callbacks() -> list:  # type: ignore
    lr_callback = LearningRateMonitor(logging_interval="epoch")

    ckpt_callback_acc = ModelCheckpoint(
        filename="epoch{epoch:03d}-val_acc-{val/acc: .3f}",
        monitor="val/acc",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        mode="max",
    )
    ckpt_callback_f1 = ModelCheckpoint(
        filename="epoch{epoch:03d}-val_f1-{val/f1: .4f}",
        monitor="val/f1",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        mode="max",
    )

    return [ckpt_callback_acc, ckpt_callback_f1, lr_callback]


def get_gpu_settings(gpu_ids: list[int], n_gpu: int):  # type: ignore
    """Get gpu settings for pytorch-lightning trainer:
    https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags
    Args:
        gpu_ids (list[int])
        n_gpu (int)
    Returns:
        tuple[str, int, str]: accelerator, devices, strategy
    """
    if not torch.cuda.is_available():
        return "cpu", None, None

    if gpu_ids is not None:
        devices = gpu_ids
        strategy = "ddp" if len(gpu_ids) > 1 else None
    elif n_gpu is not None:
        # int
        devices = n_gpu
        strategy = "ddp" if n_gpu > 1 else None
    else:
        devices = 1
        strategy = None

    return "gpu", devices, strategy


def get_trainer(arg: argparse.Namespace) -> Trainer:
    callbacks = get_basic_callbacks()
    accelerator, devices, strategy = get_gpu_settings(arg.gpu_ids, arg.n_gpu)
    my_trainer = Trainer(
        max_epochs=arg.epochs,
        callbacks=callbacks,
        default_root_dir=arg.outdir,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=True,
        deterministic=True,
        check_val_every_n_epoch=1,
        limit_train_batches=0.05,
        limit_val_batches=0.05,
        fast_dev_run=args.debug,
    )
    return my_trainer


if __name__ == "__main__":
    seed_everything(args.seed, workers=True)

    data = CreateDataloaders(
        root_dir=args.dataset,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = AnomalyClassifier(
        model_name=args.model_name, pretrained=True, num_classes=len(data.classes)
    )
    # print(ModelSummary(model))
    trainer = get_trainer(args)

    print("Args:")
    pprint(args.__dict__)
    print("Training classes:")
    pprint(data.class_to_idx)
    trainer.fit(model, data)
