import pytest
import torch

from src.distillation.comparison import PairedErrorAnalysis


def test_paired_error_analysis_reports_teacher_gap_kd_transfers_and_regressions():
    analysis = PairedErrorAnalysis(
        ("background", "small_rock"),
        trimap_kernel_size=3,
    )
    targets = torch.tensor([[[0, 0, 1], [1, 1, 0], [0, 1, 0]]])
    teacher = torch.tensor([[[0, 0, 1], [1, 0, 0], [0, 1, 0]]])
    control = torch.tensor([[[0, 1, 0], [1, 1, 0], [0, 1, 0]]])
    kd = torch.tensor([[[0, 0, 1], [1, 0, 1], [0, 1, 0]]])

    analysis.update(targets, teacher, control, kd)
    all_metrics = analysis.serializable()["all"]
    overall = all_metrics["overall"]
    small_rock = all_metrics["per_ground_truth_class"]["small_rock"]

    assert overall["pixels"] == 9
    assert overall["teacher_correct_control_wrong"] == 2
    assert overall["kd_recovers_teacher_advantage"] == 2
    assert overall["kd_corrects_control_error"] == 2
    assert overall["kd_introduces_control_error"] == 2
    assert overall["kd_recovery_rate"] == pytest.approx(1.0)
    assert overall["kd_net_accuracy_delta"] == pytest.approx(0.0)
    assert small_rock["teacher_correct_control_wrong"] == 1
    assert "trimap" in analysis.serializable()


def test_paired_error_analysis_rejects_misaligned_prediction_shapes():
    analysis = PairedErrorAnalysis(("background", "rock"))
    targets = torch.zeros((1, 2, 2), dtype=torch.long)
    with pytest.raises(ValueError, match="share"):
        analysis.update(
            targets,
            torch.zeros((1, 2, 2), dtype=torch.long),
            torch.zeros((1, 2, 2), dtype=torch.long),
            torch.zeros((1, 1, 2), dtype=torch.long),
        )
