"""Training loop for the KAT-based glaucoma classifier."""

from pathlib import Path
import csv
import os
import sys
from typing import Optional

import numpy as np
import torch
from sklearn import metrics
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics import Accuracy, AUROC, Precision, Recall
from tqdm.auto import tqdm

from dataset import Custom_Dataset
from model import kat_small_patch16_224
from polyfocalloss import PolyFocalLoss

try:
    import wandb
except ImportError:
    wandb = None


class Train:
    def __init__(self, args, feature_extract: bool = False) -> None:
        self.args = args
        self.device = torch.device(
            args.device if getattr(args, "device", "auto") != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.current_epoch = 0
        self.n_epochs = args.epochs
        self.best_acc = 0.0
        self.best_auc = 0.0
        self.best_screening_partial_auc = 0.0
        self.best_screening_sens_at_spec = 0.0
        self.feature_extract = feature_extract

        self._prepare_data()
        self._build_model()
        self._configure_optimization()
        self._restore_checkpoint()
        self._start_logging()

    def _start_logging(self) -> None:
        self.wandb_run = None
        if not getattr(self.args, "use_wandb", False):
            return
        if wandb is None:
            raise ImportError("wandb is required when --use_wandb is enabled")
        self.wandb_run = wandb.init(
            project=self.args.wandb_project,
            name=self.args.log_name,
            config=vars(self.args),
        )

    def _prepare_data(self) -> None:
        train_set = Custom_Dataset(
            self.args.train_input, classes=self.args.classes, is_train=True,
            image_size=self.args.image_size,
        )
        val_set = Custom_Dataset(
            self.args.valid_input, classes=self.args.classes, is_train=False,
            image_size=self.args.image_size,
        )

        class_counts = np.bincount(
            [label for _, label in train_set.samples],
            minlength=len(self.args.classes),
        )
        class_weights = np.zeros_like(class_counts, dtype=np.float64)
        nonzero = class_counts > 0
        class_weights[nonzero] = 1.0 / class_counts[nonzero]
        sample_weights = torch.as_tensor(
            [class_weights[label] for _, label in train_set.samples], dtype=torch.double
        )
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )

        workers = self.args.num_workers
        self.train_loader = DataLoader(
            train_set, batch_size=self.args.batch_size, sampler=sampler,
            num_workers=workers, pin_memory=self.device.type == "cuda",
        )
        self.val_loader = DataLoader(
            val_set, batch_size=self.args.batch_size, shuffle=False,
            num_workers=workers, pin_memory=self.device.type == "cuda",
        )

    def _build_model(self) -> None:
        self.model = kat_small_patch16_224().to(self.device)
        if self.feature_extract:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def _configure_optimization(self) -> None:
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = Adam(trainable, lr=self.args.learning_rate)
        total_steps = max(1, len(self.train_loader) * self.n_epochs)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)
        self.criterion = PolyFocalLoss().to(self.device)
        n_classes = len(self.args.classes)
        self.precision = Precision(task="multiclass", average="macro", num_classes=n_classes).to(self.device)
        self.accuracy = Accuracy(task="multiclass", num_classes=n_classes).to(self.device)
        self.recall = Recall(task="multiclass", average="macro", num_classes=n_classes).to(self.device)
        self.auc = AUROC(task="multiclass", average="macro", num_classes=n_classes).to(self.device)

    @property
    def checkpoint_dir(self) -> Path:
        path = Path(self.args.checkpoint_folder)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _restore_checkpoint(self) -> None:
        path = self.checkpoint_dir / "last_model.pth"
        if not path.exists():
            print("No previous checkpoint found; starting a new run.")
            return
        state = torch.load(path, map_location="cpu")
        self.current_epoch = int(state.get("step", 0))
        self.best_auc = float(state.get("best_auc", 0.0))
        self.best_acc = float(state.get("best_accuracy", 0.0))
        self.model.load_state_dict(state["generator_state_dict"])
        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        print(f"Restored checkpoint from epoch {self.current_epoch}.")

    @staticmethod
    def _binary_scores(probabilities: torch.Tensor) -> torch.Tensor:
        top_prob, predicted = probabilities.max(dim=-1)
        return torch.where(predicted == 1, top_prob, 1 - top_prob)

    def screening_partial_auc(self, probabilities: torch.Tensor,
                              labels: torch.Tensor, min_spec: float = 0.90) -> torch.Tensor:
        scores = self._binary_scores(probabilities).detach().cpu().numpy()
        targets = labels.detach().cpu().numpy().astype(bool)
        try:
            value = metrics.roc_auc_score(targets, scores, max_fpr=1 - min_spec)
        except ValueError:
            value = 0.0
        return torch.tensor(value, dtype=torch.float32, device=self.device)

    def screening_sens_at_spec(self, probabilities: torch.Tensor,
                               labels: torch.Tensor, at_spec: float = 0.95) -> torch.Tensor:
        scores = self._binary_scores(probabilities).detach().cpu().numpy()
        targets = labels.detach().cpu().numpy().astype(bool)
        try:
            fpr, tpr, _ = metrics.roc_curve(targets, scores, drop_intermediate=False)
            sensitivity = tpr[(1 - fpr) >= at_spec]
            value = float(sensitivity[-1]) if len(sensitivity) else 0.0
        except ValueError:
            value = 0.0
        return torch.tensor(value, dtype=torch.float32, device=self.device)

    def _reset_metrics(self) -> None:
        for metric in (self.precision, self.accuracy, self.recall, self.auc):
            metric.reset()

    def _record(self, split: str, values: dict) -> None:
        csv_path = self.checkpoint_dir / f"results_{split}.csv"
        new_file = not csv_path.exists()
        with csv_path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(["epoch", *values.keys()])
            writer.writerow([self.current_epoch, *values.values()])

        if self.wandb_run is not None:
            wandb.log({f"{split}_{key}": value for key, value in values.items()})

    def train_epoch(self) -> None:
        self.model.train()
        for _, inputs, labels in tqdm(self.train_loader, desc=f"Train {self.current_epoch}"):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            probabilities = outputs.softmax(dim=1)
            self.precision.update(probabilities, labels)
            self.accuracy.update(probabilities, labels)
            self.recall.update(probabilities, labels)
            self.auc.update(probabilities, labels)

        values = {
            "accuracy": self.accuracy.compute().item(),
            "precision": self.precision.compute().item(),
            "recall": self.recall.compute().item(),
            "auc": self.auc.compute().item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        self._record("train", values)
        self._reset_metrics()
        print(f"Epoch {self.current_epoch}: train {values}")

    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        all_probabilities, all_labels = [], []

        for _, inputs, labels in tqdm(self.val_loader, desc=f"Valid {self.current_epoch}"):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            probabilities = self.model(inputs).softmax(dim=1)
            all_probabilities.append(probabilities)
            all_labels.append(labels)
            self.precision.update(probabilities, labels)
            self.accuracy.update(probabilities, labels)
            self.recall.update(probabilities, labels)
            self.auc.update(probabilities, labels)

        probabilities = torch.cat(all_probabilities)
        labels = torch.cat(all_labels)
        values = {
            "accuracy": self.accuracy.compute().item(),
            "precision": self.precision.compute().item(),
            "recall": self.recall.compute().item(),
            "auc": self.auc.compute().item(),
            "partial_auc": self.screening_partial_auc(probabilities, labels).item(),
            "sensitivity_at_spec": self.screening_sens_at_spec(probabilities, labels).item(),
        }
        self._record("val", values)
        self._reset_metrics()
        print(f"Epoch {self.current_epoch}: validation {values}")
        return values

    def save_checkpoint(self, name: str = "last") -> None:
        state = {
            "step": self.current_epoch,
            "best_accuracy": self.best_acc,
            "best_auc": self.best_auc,
            "generator_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(state, self.checkpoint_dir / f"{name}_model.pth")

    def run(self) -> None:
        for epoch in range(self.current_epoch + 1, self.n_epochs + 1):
            self.current_epoch = epoch
            self.train_epoch()
            results = self.val_epoch()

            if results["accuracy"] > self.best_acc:
                self.best_acc = results["accuracy"]
                self.save_checkpoint("best_acc")
            if results["auc"] > self.best_auc:
                self.best_auc = results["auc"]
                self.save_checkpoint("best_auc")
            if results["partial_auc"] > self.best_screening_partial_auc:
                self.best_screening_partial_auc = results["partial_auc"]
                self.save_checkpoint("best_screening_partial_auc")
            if results["sensitivity_at_spec"] > self.best_screening_sens_at_spec:
                self.best_screening_sens_at_spec = results["sensitivity_at_spec"]
                self.save_checkpoint("best_screening_sens_at_spec")
            self.save_checkpoint("last")
