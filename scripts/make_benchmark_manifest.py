"""Write a split manifest for Cityscapes or ADE20K.

Neither benchmark publishes test labels, so the official validation set is the
held-out test split and a seeded fraction of the official training set is used
for checkpoint selection. Standard library only, so it runs on a login node:

    python3 scripts/make_benchmark_manifest.py cityscapes /datasets/cityscapes \
        experiments/sard/cityscapes/split_seed42.json
"""

import argparse
import json
import random
from pathlib import Path


def _cityscapes_ids(root: Path, split: str) -> list[str]:
    suffix = "_leftImg8bit.png"
    base = root / "leftImg8bit"
    return sorted(
        str(path.relative_to(base).as_posix())[: -len(suffix)]
        for path in (base / split).glob(f"*/*{suffix}")
    )


def _ade20k_ids(root: Path, split: str) -> list[str]:
    base = root / "ADEChallengeData2016" / "images"
    return sorted(
        str(path.relative_to(base).with_suffix("").as_posix())
        for path in (base / split).glob("*.jpg")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("cityscapes", "ade20k"))
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset == "cityscapes":
        trainval, test = _cityscapes_ids(args.dataset_root, "train"), _cityscapes_ids(
            args.dataset_root, "val"
        )
    else:
        trainval, test = _ade20k_ids(args.dataset_root, "training"), _ade20k_ids(
            args.dataset_root, "validation"
        )
    if not trainval or not test:
        raise SystemExit(f"no {args.dataset} samples found under {args.dataset_root}")

    shuffled = list(trainval)
    random.Random(args.seed).shuffle(shuffled)
    validation_count = round(len(shuffled) * args.validation_fraction)
    validation = sorted(shuffled[:validation_count])
    train = sorted(shuffled[validation_count:])

    payload = {
        "schema_version": 1,
        "dataset": args.dataset,
        "seed": args.seed,
        "excluded_id_files": [],
        "included_id_files": [],
        "counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "total": len(train) + len(validation) + len(test),
        },
        "splits": {"train": train, "validation": validation, "test": test},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, indent=1)
        file.write("\n")
    print(json.dumps(payload["counts"]))


if __name__ == "__main__":
    main()
