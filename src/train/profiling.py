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


class NvtxRangesCallback(Callback):
    """Add coarse, balanced NVTX ranges around Lightning lifecycle hooks.

    This keeps trace instrumentation outside the model and its training steps.
    Range names intentionally omit batch indices, which keeps an Nsight Systems
    timeline readable even for long runs.
    """

    def __init__(self) -> None:
        self._ranges: list[str] = []

    def _push(self, name: str) -> None:
        if not torch.cuda.is_available():
            return
        torch.cuda.nvtx.range_push(name)
        self._ranges.append(name)

    def _pop(self, expected: str) -> None:
        if not self._ranges or self._ranges[-1] != expected:
            return
        torch.cuda.nvtx.range_pop()
        self._ranges.pop()

    def _close_all(self) -> None:
        while self._ranges:
            torch.cuda.nvtx.range_pop()
            self._ranges.pop()

    def on_fit_start(self, trainer: Trainer, pl_module) -> None:
        self._push("seg-opt/fit")

    def on_fit_end(self, trainer: Trainer, pl_module) -> None:
        self._close_all()

    def on_exception(self, trainer: Trainer, pl_module, exception: BaseException) -> None:
        self._close_all()

    def on_train_epoch_start(self, trainer: Trainer, pl_module) -> None:
        self._push("seg-opt/train_epoch")

    def on_train_epoch_end(self, trainer: Trainer, pl_module) -> None:
        self._pop("seg-opt/train_epoch")

    def on_train_batch_start(self, trainer: Trainer, pl_module, batch, batch_idx: int) -> None:
        self._push("seg-opt/train_batch")

    def on_train_batch_end(self, trainer: Trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        self._pop("seg-opt/train_batch")

    def on_validation_epoch_start(self, trainer: Trainer, pl_module) -> None:
        self._push("seg-opt/validation_epoch")

    def on_validation_epoch_end(self, trainer: Trainer, pl_module) -> None:
        self._pop("seg-opt/validation_epoch")

    def on_validation_batch_start(
        self, trainer: Trainer, pl_module, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._push("seg-opt/validation_batch")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._pop("seg-opt/validation_batch")

    def on_test_epoch_start(self, trainer: Trainer, pl_module) -> None:
        self._push("seg-opt/test_epoch")

    def on_test_epoch_end(self, trainer: Trainer, pl_module) -> None:
        self._pop("seg-opt/test_epoch")

    def on_test_batch_start(
        self, trainer: Trainer, pl_module, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._push("seg-opt/test_batch")

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._pop("seg-opt/test_batch")


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
