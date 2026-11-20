# Lunar logit distillation, 80 epochs

Students: `segformer_b0` (`nvidia/mit-b0`),
`segformer_b1` (`nvidia/mit-b1`),
and `segformer_b2` (`nvidia/mit-b2`).
Three seeds (42/43/44) run vanilla logit KD for 80 epochs on the fixed clean/aligned Lunar split
with the existing seed-42 B4 teacher. Each run uses one H100; the array permits
three concurrent GPU jobs. The effective batch size is 16; KD temperature and
weight are 1. Checkpoints are selected by validation mIoU.

Run from the deployed source directory on Eagle:
```bash
export IMAGE_PATH="$HOME/pl1200-01/project_data/containers/seg-opt/development/current.sif"
export DATA_PATH="$HOME/pl1200-01/project_data/datasets"
export OUTPUT_PATH="$HOME/pl1200-01/project_data/results"
export WANDB_ENTITY=ogrodowskijedrzej-put
export WANDB_PROJECT=seg-opt
```

Use an existing W&B login, `WANDB_API_KEY`, or `WANDB_API_KEY_FILE` (private, single-line file).

An online smoke run gates the three-seed array through `afterok`. Campaign IDs cannot be reused.
SLURM stdout/stderr and job IDs in `submission.txt` are saved in the campaign directory.
Runs save configuration, split, metrics, best checkpoint, train log and local W&B files.
The teacher must exist at `$OUTPUT_PATH/sard/teacher_segformer_b4/checkpoints/best.ckpt`.
Model checkpoint files remain on HPC (`log_model=False`).
