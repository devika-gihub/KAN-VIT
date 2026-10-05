# KAT Retinal Classification Code

This repository contains the training, evaluation, and explainability code used for a retinal-image classification experiment.

## Important attribution

The `model.py` file contains the KAT architecture supplied with the original project and explicitly attributes the architecture to Xingyi Yang. The `efficient_kan.py` file also contains an implementation supplied with the project. Before public redistribution, retain the original upstream copyright/license and citation information for these components.

The surrounding dataset, training, evaluation, and visualization scripts have been reorganized for readability, configuration, and reproducibility. Refactoring does not change the scientific ownership of third-party architecture code.

## Main files

- `main.py` — command-line training entry point.
- `dataset.py` — dataset discovery and preprocessing.
- `train.py` — training, validation, checkpointing, and metrics.
- `test.py` — evaluation of saved checkpoints.
- `gradcam.py` / `gradcam1.py` — Grad-CAM utilities.
- `utils.py` — visualization and layer-selection helpers.
- `losses.py` / `polyfocalloss.py` — loss functions.
- `model.py` — KAT model implementation supplied with the project.
- `efficient_kan.py` — Efficient-KAN implementation supplied with the project.

## Security change

The original training script contained a hard-coded Weights & Biases credential. That credential has intentionally been removed. Use `wandb login` or the normal `WANDB_API_KEY` environment variable outside the source code.

## Training example

```bash
python main.py \
  --train_input /workspace/ARD_clahe/train \
  --valid_input /workspace/ARD_clahe/valid \
  --checkpoint_folder /workspace/checkpoint \
  --epochs 1000 \
  --batch_size 16
```

Add `--use_wandb` only when experiment logging is required.

## Testing example

```bash
python test.py \
  --data /workspace/ARD_clahe/ACRIMA \
  --checkpoint_dir /workspace/checkpoint \
  --output_dir /workspace/evaluation
```

## Reproducibility note

Record the Python version, PyTorch/CUDA versions, dataset split, random seed, preprocessing configuration, model checkpoint, and exact command line used for each reported experiment.
