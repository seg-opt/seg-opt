# Experiment map

## Clean baseline benchmark

[`baselines_benchmark/`](baselines_benchmark/) compares the Fast-SCNN student
with four independent teacher candidates on clean masks and the aligned
seed-42 split. Run its smoke check, then its five-job array, and use its summary
to choose a teacher before doing any distillation.

## CAP ablation

[`cap/`](cap/) compares the SegFormer-B4 to SegFormer-B0 vanilla logit-KD
control with the same run plus the ScaleKD cross-attention feature loss. Its
submitter runs both smoke paths first and releases both full jobs only after
that check succeeds. The trained clean SegFormer-B4 checkpoint is required.

## Deferred: BPKD

[`bpkd/`](bpkd/) contains the prepared BPKD experiment matrix. Do not submit
it until the clean baseline benchmark establishes a positive teacher margin.

## Exploratory work

[`dinov3_demo/`](dinov3_demo/) and [`dinov3_probe_v2/`](dinov3_probe_v2/) are
standalone DINOv3 exploration, not part of the benchmark decision path.
