# SegFormer-B4 to SegFormer-B0 CAP ablation

Both runs use the same student initialization, clean seed-42 split, task loss,
and vanilla logit distillation. The treatment adds only the final-stage CAP
feature loss with the ScaleKD spatial-loss weight `0.08`.

The smoke job runs one batch through both configurations. If both checks pass,
the submitter releases the no-CAP control and CAP treatment in parallel:

```bash
bash experiments/cap/submit.sh
```

`IMAGE_PATH` and `DATA_PATH` must point to the development container and dataset
directory, as for the other experiment submitters. The trained B4 checkpoint
must already exist at
`results/baselines_benchmark/segformer_b4/seed42/checkpoints/best.ckpt`.
