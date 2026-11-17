"""Fail before training if authenticated W&B access is unavailable."""

import os

import wandb


if os.environ.get("WANDB_MODE") != "online":
    raise RuntimeError("This campaign requires WANDB_MODE=online")
if not wandb.Api(timeout=30).viewer:
    raise RuntimeError("W&B authentication is required before training")
print("Authenticated W&B access verified.", flush=True)
