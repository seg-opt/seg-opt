# SegFormer-B4 to SegFormer-B2: Lunar, 80 epochs, three seeds

Student: `segformer_b2` (`nvidia/mit-b2`).

See the [80-epoch campaign guide](../b4_lunar_80e.md) for setup and training settings.

```bash
bash experiments/logit_distillation/b4_to_b2_lunar_80e/submit.sh
```

Results:
```text
~/pl1200-01/project_data/results/logit_distillation/b4_to_b2_lunar_80e/<campaign-id>/
  smoke/
  seed42/
  seed43/
  seed44/
```
