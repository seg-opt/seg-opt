import pytest

from tests.unit.test_submit_hydra import load_hydra_config


@pytest.mark.parametrize(
    ("dataset", "teacher"),
    [
        ("lunar", "benchmark_segformer_b4"),
        ("ade20k", "ade20k_segformer_b4"),
        ("cityscapes", "cityscapes_segformer_b4"),
    ],
)
@pytest.mark.parametrize("seed", [42, 43, 44])
def test_seeds_alter_training_without_altering_dataset(dataset, teacher, seed, tmp_path):
    overrides = (f"dataset={dataset}", "method=sard", f"teacher={teacher}", "student=fast_scnn")
    default = load_hydra_config(tmp_path, *overrides)
    changed = load_hydra_config(tmp_path, *overrides, f"training.seed={seed}")
    assert default.training.seed == 42
    assert changed.training.seed == seed
    assert changed.data == default.data
