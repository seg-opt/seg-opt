# SegFormer-B4 to SegFormer-B1: Lunar, 80 epochs, three seeds

Three independent vanilla logit-KD runs use seeds 42/43/44, 80 epochs,
the fixed clean/aligned Lunar split, and the existing seed-42 B4 teacher.
Each run uses one H100; the full array permits three concurrent GPU jobs.
The effective batch size is 16, KD temperature and weight are 1, and the
checkpoint is selected by validation mIoU.

Run from the deployed source directory on Eagle:

```bash
export IMAGE_PATH="$HOME/pl1200-01/project_data/containers/seg-opt/development/current.sif"
export DATA_PATH="$HOME/pl1200-01/project_data/datasets"
export OUTPUT_PATH="$HOME/pl1200-01/project_data/results"
export WANDB_ENTITY=ogrodowskijedrzej-put
export WANDB_PROJECT=seg-opt

# Use an existing wandb login, or enter a key without displaying it.
read -rsp 'W&B API key: ' WANDB_API_KEY
echo
export WANDB_API_KEY

bash experiments/logit_distillation/b4_to_b1_lunar_80e/submit.sh
```

Alternatively, set `WANDB_API_KEY_FILE` to a private, single-line key file.
Never commit credentials. The job passes authentication into the container,
verifies authenticated API access, and forces online logging. A separate
online smoke run gates the full array through an `afterok` dependency.
Check its W&B run link in the SLURM log.

All runs save configuration, split, metrics, best checkpoint, train log,
and local W&B files under:

```text
~/pl1200-01/project_data/results/logit_distillation/b4_to_b1_lunar_80e/<campaign-id>/
  smoke/
  seed42/
  seed43/
  seed44/
```

SLURM stdout/stderr and submission IDs are saved in the campaign directory.
Reusing a campaign ID is rejected. W&B receives metrics and hyperparameters;
model checkpoint files remain on HPC (`log_model=False`). The teacher must
exist at `$OUTPUT_PATH/sard/teacher_segformer_b4/checkpoints/best.ckpt`.
Deploy source files with Unix line endings, including the split manifest,
whose checksum is part of teacher/student provenance validation.
