from scripts.summarize_baselines_benchmark import (
    STUDENT,
    TEACHER_CANDIDATES,
    render_summary,
)


def _metrics(name: str, validation_miou: float, validation_rock_iou: float):
    def split(miou: float, rock_iou: float):
        return {
            "miou": miou,
            "per_class": {"small_rock": {"iou": rock_iou}},
            "trimap": {"per_class": {"small_rock": {"iou": rock_iou - 0.1}}},
        }

    return {
        "model": {"name": name},
        "data": {"mask_variant": "clean"},
        "run_kind": "full",
        "validation": split(validation_miou, validation_rock_iou),
        "test": split(validation_miou - 0.01, validation_rock_iou - 0.01),
    }


def test_summary_selects_by_validation_miou_and_reports_student_margin():
    metrics = {STUDENT: _metrics(STUDENT, 0.40, 0.30)}
    metrics.update(
        {
            name: _metrics(name, 0.50 + index * 0.01, 0.31 + index * 0.01)
            for index, name in enumerate(TEACHER_CANDIDATES)
        }
    )

    summary = render_summary(metrics)

    assert "Selected teacher: **mask2former_swinl**" in summary
    assert "Validation margin over Fast-SCNN: **+0.1500 mIoU**" in summary
    assert "There is a task-quality margin worth testing with distillation." in summary
