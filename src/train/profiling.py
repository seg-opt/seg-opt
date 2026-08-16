from pathlib import Path

import torch
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.profilers import PyTorchProfiler


class GpuStatsCallback(Callback):
    def __init__(self, every_n_steps: int = 20) -> None:
        self.every_n_steps = every_n_steps

    def on_train_start(self, trainer: Trainer, pl_module) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not torch.cuda.is_available() or trainer.global_step % self.every_n_steps:
            return

        device = pl_module.device
        gib = 1024**3
        metrics = {
            "gpu/memory_allocated_gib": torch.cuda.memory_allocated(device) / gib,
            "gpu/memory_reserved_gib": torch.cuda.memory_reserved(device) / gib,
            "gpu/max_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
        }
        try:
            metrics["gpu/utilization_percent"] = float(torch.cuda.utilization(device))
        except (AttributeError, ModuleNotFoundError, RuntimeError):
            pass
        pl_module.log_dict(metrics, on_step=True, on_epoch=False, logger=True)


def make_profiler(profile_dir: str | Path) -> PyTorchProfiler:
    profile_dir = Path(profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    return PyTorchProfiler(
        dirpath=profile_dir,
        filename="training",
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        group_by_input_shapes=True,
        export_to_chrome=True,
        row_limit=50,
    )
