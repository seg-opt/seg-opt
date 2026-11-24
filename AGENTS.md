# Repository rules

- Do not write `from __future__ import annotations` in Python files.
- Define new training jobs with the shared Hydra config in `experiments/hydra`.
  Select the dataset, method, teacher, and student explicitly. Reuse presets and
  pass seeds and other hyperparameter overrides on the command line.
- Submit from the Slurm login host with the shared runner. It saves the composed
  YAML and overrides in the snapshot, submits the smoke job with W&B offline,
  and submits the full job with W&B online and an `afterok` dependency.
  Each stage writes to its own output directory.

  ```bash
  OUTPUT_PATH="$HOME/pl1200-01/project_data/results" \
    uv run --only-group submit python -m scripts.experiment submit-hydra \
    dataset=lunar method=vanilla teacher=lunar_segformer_b4 student=segformer_b2 \
    training.seed=43
  ```

  Use an output path or cluster profile that contains the selected teacher checkpoint.
