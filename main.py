"""Command-line entry point for KAT glaucoma classification training."""

import argparse
import os

from train import Train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the retinal-image classifier.")
    parser.add_argument("--checkpoint_folder", default="/workspace/checkpoint")
    parser.add_argument("--log_name", default="kat_small_retinal_classifier")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a CUDA device")
    parser.add_argument("--train_input", default="/workspace/ARD_clahe/train")
    parser.add_argument("--valid_input", default="/workspace/ARD_clahe/valid")
    parser.add_argument("--classes", nargs="+", default=["NRG", "RG"])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="class_glaucoma_aug")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    Train(args).run()


if __name__ == "__main__":
    main()
