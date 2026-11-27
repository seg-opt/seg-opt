# Repository rules

- Do not write `from __future__ import annotations` in Python files.
- Define training, transfer, and evaluation jobs with the shared Hydra config in `experiments/hydra`.
  Select the dataset, method, teacher, and student explicitly. Reuse presets and
  pass seeds and other hyperparameter overrides on the command line.
- Submit from the Slurm login host with the Bash launcher. It passes the same
  composed YAML to a smoke job with W&B offline and a full job with W&B online
  and an `afterok` dependency. Each job saves its configuration in its results.
  Keep the checkout stable until both jobs finish.

  ```bash
  OUTPUT_PATH="/path/to/writable/results" \
    scripts/submit \
    dataset=lunar method=vanilla teacher=lunar_segformer_b4 student=segformer_b2 \
    training.seed=43
  ```

  Use a writable output path that contains the selected teacher checkpoint.
  See `experiments/README.md` for launcher options and named campaigns.
