import argparse
import json
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "experiments/hydra"


def compose_experiment(overrides=()):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    def resolve(arguments):
        return OmegaConf.to_container(
            compose(config_name="config", overrides=list(arguments)),
            resolve=True,
            throw_on_missing=True,
        )

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        config = resolve(overrides)
        teacher_overrides = config.pop("_teacher_overrides", None)
        teacher = None
        if teacher_overrides and config["distillation"]["teacher"]["config_path"] == "teacher_config.yaml":
            teacher = resolve(teacher_overrides)
    distillation = config.get("distillation")
    if distillation is not None and (
        "method" not in distillation or "teacher" not in distillation
    ):
        raise ValueError("select teacher=none for supervised training and a teacher for distillation")
    return config, teacher


def write_configs(config, teacher, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if teacher is not None:
        teacher_path = path.with_suffix(".teacher.yaml")
        with teacher_path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(teacher, stream, sort_keys=False)
        config = {**config, "distillation": {
            **config["distillation"], "teacher": {
                **config["distillation"]["teacher"], "config_path": str(teacher_path.resolve()),
            },
        }}
    with path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    return path


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bundle", action="store_true")
    mode.add_argument("--write-config", type=Path)
    mode.add_argument("--unpack", type=Path)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if args.unpack:
        config, teacher = json.loads(args.overrides[0])
        write_configs(config, teacher, args.unpack)
        return

    config, teacher = compose_experiment(args.overrides)
    if args.bundle:
        print(json.dumps([config, teacher], separators=(",", ":"), allow_nan=False))
    else:
        write_configs(config, teacher, args.write_config)


if __name__ == "__main__":
    main()
