"""Evaluate saved checkpoints on an ImageFolder test set."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, AUROC, CohenKappa, F1Score, Precision, Recall, Specificity
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from model import kat_small_patch16_224


CHECKPOINTS = {
    "best_acc": "accuracy.png",
    "best_auc": "auc.png",
    "best_screening_partial_auc": "partial_auc.png",
    "best_screening_sens_at_spec": "sensitivity.png",
    "last": "last.png",
}


def build_metrics(device: torch.device):
    return {
        "precision": Precision(task="multiclass", average="macro", num_classes=2).to(device),
        "accuracy": Accuracy(task="multiclass", num_classes=2).to(device),
        "recall": Recall(task="multiclass", average="macro", num_classes=2).to(device),
        "auc": AUROC(task="multiclass", average="macro", num_classes=2).to(device),
        "kappa": CohenKappa(task="multiclass", num_classes=2).to(device),
        "specificity": Specificity(task="multiclass", num_classes=2).to(device),
        "f1": F1Score(task="multiclass", num_classes=2).to(device),
    }


def evaluate(model, loader, device):
    metrics = build_metrics(device)
    y_true, y_pred = [], []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            probabilities = logits.softmax(dim=1)
            predictions = probabilities.argmax(dim=1)

            for metric in metrics.values():
                metric.update(probabilities, labels)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    values = {name: metric.compute().item() for name, metric in metrics.items()}
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return values, cm


def save_confusion_matrix(cm, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KAT checkpoints.")
    parser.add_argument("--data", default="/workspace/ARD_clahe/ACRIMA")
    parser.add_argument("--checkpoint_dir", default="/workspace/checkpoint")
    parser.add_argument("--output_dir", default="/workspace/evaluation")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = datasets.ImageFolder(args.data, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = kat_small_patch16_224().to(device)
    checkpoint_dir = Path(args.checkpoint_dir)

    for name, figure_name in CHECKPOINTS.items():
        checkpoint = checkpoint_dir / f"{name}_model.pth"
        if not checkpoint.exists():
            print(f"Skipping missing checkpoint: {checkpoint}")
            continue

        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["generator_state_dict"])
        values, cm = evaluate(model, loader, device)
        save_confusion_matrix(cm, output_dir / figure_name)
        print(f"\n{name} checkpoint")
        for key, value in values.items():
            print(f"  {key:12s}: {value:.6f}")


if __name__ == "__main__":
    main()
