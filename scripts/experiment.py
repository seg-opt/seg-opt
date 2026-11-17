from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid


TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED", "BLOCKED"}
ALLOWLIST = ("src", "scripts", "experiments", "tests", "pyproject.toml", "uv.lock")
RESOURCES = {"partition", "constraint", "gpus-per-node", "cpus-per-task", "mem", "time", "ntasks"}


def parse_seed(value):
    try:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError
        seed = int(value)
        if not 0 <= seed <= 4294967295:
            raise ValueError
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("seed must be an unsigned 32-bit integer") from None
    return seed


def build_parser():
    parser = argparse.ArgumentParser(description="Snapshot-owned experiment execution")
    commands = parser.add_subparsers(dest="command", required=True)
    submit_parser = commands.add_parser("submit")
    submit_parser.add_argument("configs", nargs="+")
    pipeline = commands.add_parser("pipeline")
    pipeline.add_argument("--config", required=True)
    pipeline.add_argument("--baseline-artifacts", required=True)
    teacher = pipeline.add_mutually_exclusive_group(required=True)
    teacher.add_argument("--teacher-config")
    teacher.add_argument("--teacher-artifacts")
    pipeline.add_argument("--after-job")
    for command in (submit_parser, pipeline):
        command.add_argument("--profile", type=Path)
        command.add_argument("--wandb-mode", choices=("offline", "online", "disabled"))
        command.add_argument("--seed", type=parse_seed)
    for name in ("status", "retry"):
        commands.add_parser(name).add_argument("run", type=Path)
    commands.add_parser("compare").add_argument("artifacts", nargs="+", type=Path)
    launch_parser = commands.add_parser("launch")
    launch_parser.add_argument("--smoke", action="store_true")
    launch_parser.add_argument("configs", nargs="+")
    worker_parser = commands.add_parser("worker")
    worker_parser.add_argument("manifest", type=Path)
    worker_parser.add_argument("stage")
    worker_parser.add_argument("--container", action="store_true")
    return parser


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".record-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository():
    return Path(os.environ.get("REPO_PATH", Path(__file__).resolve().parents[1])).resolve()


def inside(path, parent):
    try:
        return Path(path).resolve().relative_to(Path(parent).resolve())
    except ValueError:
        raise ValueError(f"{path} must be inside {parent}") from None


def safe_config(value, root):
    path = Path(value).expanduser()
    path = path if path.is_absolute() else root / path
    relative = inside(path, root)
    if relative.parts[0] not in ALLOWLIST:
        raise ValueError(f"config must be in the source snapshot allowlist: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    if any(excluded(part) for part in relative.parts):
        raise ValueError(f"config excluded from source snapshot: {path}")
    return relative.as_posix()


def excluded(name):
    return (name.startswith(".") or name in {"results", "cache", "caches", "wandb", "__pycache__",
                                            "venv", "env", "node_modules", "secrets", "credentials",
                                            "credentials.json", "id_rsa", "id_ed25519"}
            or name.endswith((".pyc", ".pyo", ".sif", ".pem", ".key")))


def snapshot(root, destination):
    destination.mkdir()
    hashes = {}

    def copy(source, target, ancestors=()):
        resolved = source.resolve()
        relative = inside(resolved, root)
        if not relative.parts or relative.parts[0] not in ALLOWLIST:
            raise ValueError(f"snapshot symlink targets non-allowlisted content: {source}")
        if resolved in ancestors:
            raise ValueError(f"cyclic snapshot symlink: {source}")
        if any(excluded(part) for part in resolved.relative_to(root).parts):
            raise ValueError(f"snapshot symlink targets excluded content: {source}")
        if source.is_dir():
            target.mkdir()
            for child in sorted(source.iterdir()):
                if not excluded(child.name):
                    copy(child, target / child.name, (*ancestors, resolved))
        elif source.is_file():
            shutil.copyfile(source, target)
            hashes[target.relative_to(destination).as_posix()] = sha256(target)
        else:
            raise ValueError(f"unsupported snapshot entry: {source}")

    for name in ALLOWLIST:
        source = root / name
        if source.exists() or source.is_symlink():
            copy(source, destination / name)
    return hashes


def load_profile(args, root):
    raw = read_json(args.profile or root / "scripts/clusters/eagle.json")
    keys = ("account", "image_path", "data_path", "output_path", "cache_path", "wandb_mode",
            "wandb_project", "gpu", "smoke", "cpu")
    if set(raw) - set(keys):
        raise ValueError("unsupported profile fields; profiles must contain only scheduler/storage/W&B defaults")
    profile = {key: raw[key] for key in keys if key in raw}
    for kind in ("gpu", "smoke", "cpu"):
        if not isinstance(profile.get(kind), dict) or set(profile[kind]) - RESOURCES:
            raise ValueError(f"unsupported scheduler resource keys in {kind}")
    for key in ("image_path", "data_path", "output_path", "cache_path"):
        value = os.environ.get(key.upper(), profile[key])
        value = os.path.expandvars(value)
        if "$" in value:
            raise ValueError(f"unexpanded environment variable in {key}")
        profile[key] = str(Path(value).expanduser().resolve())
    profile["wandb_mode"] = args.wandb_mode or os.environ.get("WANDB_MODE", profile.get("wandb_mode", "offline"))
    profile["wandb_project"] = os.environ.get("WANDB_PROJECT", profile.get("wandb_project", "seg-opt"))
    if profile["wandb_mode"] not in {"offline", "online", "disabled"}:
        raise ValueError("invalid WANDB_MODE")
    if not Path(profile["image_path"]).is_file():
        raise FileNotFoundError(profile["image_path"])
    if not Path(profile["data_path"]).is_dir():
        raise FileNotFoundError(profile["data_path"])
    return profile


def external_artifact(value, output):
    path = Path(value).expanduser().resolve()
    try:
        inside(path, output)
    except ValueError:
        raise ValueError(f"artifact {path} is outside OUTPUT_PATH; set OUTPUT_PATH to the common results root") from None
    return str(path)


def container_path(manifest, value):
    path = Path(value).resolve()
    for host, container in ((manifest["profile"]["output_path"], "/output"),
                            (manifest["snapshot"], "/workspace")):
        try:
            return str(Path(container) / path.relative_to(Path(host)))
        except ValueError:
            pass
    raise ValueError(f"path is not container-bound: {value}; check OUTPUT_PATH")


def stage(config=None, needs=(), kind="gpu", smoke=False):
    return {"config": config, "needs": list(needs), "kind": kind, "smoke": smoke}


def submit(args):
    root = repository()
    profile = load_profile(args, root)
    stages = {}
    artifacts = {}
    if args.command == "pipeline":
        config = safe_config(args.config, root)
        artifacts["baseline"] = external_artifact(args.baseline_artifacts, profile["output_path"])
        if args.after_job and (args.teacher_config or not re.fullmatch(r"[1-9][0-9]*", args.after_job)):
            raise ValueError("--after-job requires adopted teacher artifacts and a positive job ID")
        if args.teacher_config:
            teacher = safe_config(args.teacher_config, root)
            stages["teacher-smoke"] = stage(teacher, smoke=True)
            stages["teacher"] = stage(teacher, ["teacher-smoke"])
        else:
            artifacts["teacher"] = external_artifact(args.teacher_artifacts, profile["output_path"])
        stages["gate"] = stage(needs=["teacher"] if args.teacher_config else [], kind="cpu")
        stages["kd-smoke"] = stage(config, ["gate"], smoke=True)
        stages["kd"] = stage(config, ["kd-smoke"])
    else:
        configs = [safe_config(value, root) for value in args.configs]
        names = [Path(config).stem for config in configs]
        if len(set(names)) != len(names) or any(name == "summary" or name.endswith("-smoke") for name in names):
            raise ValueError("config stems must be unique and not use reserved stage names")
        for name, config in zip(names, configs):
            stages[name + "-smoke"] = stage(config, smoke=True)
            stages[name] = stage(config, [name + "-smoke"])
    seed = getattr(args, "seed", None)
    if seed is not None:
        for definition in stages.values():
            if definition["kind"] == "gpu":
                definition["seed"] = seed
    stages["summary"] = stage(needs=list(stages), kind="cpu")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:12]
    run = Path(profile["output_path"]) / "experiments" / run_id
    run.mkdir(parents=True, exist_ok=False)
    (run / "logs").mkdir()
    (run / "artifacts").mkdir()
    hashes = snapshot(root, run / "source")
    manifest = {"schema_version": 1, "id": run_id, "run": str(run), "snapshot": str(run / "source"),
                "profile": profile, "source_hashes": hashes,
                "image": {"path": profile["image_path"], "sha256": sha256(profile["image_path"])},
                "stages": stages, "artifacts": artifacts, "external_job": getattr(args, "after_job", None)}
    atomic_json(run / "manifest.json", manifest)
    with run_lock(run):
        schedule(manifest, list(stages))
    return str(run)


@contextmanager
def run_lock(run):
    with (Path(run) / ".submission.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def attempts(run, name):
    return sorted((Path(run) / "stages" / name).glob("attempt-*/attempt.json"))


def latest_attempt(run, name):
    records = attempts(run, name)
    return records[-1].parent if records else None


def execute(command, **kwargs):
    return subprocess.run(command, check=True, text=True, capture_output=True, **kwargs)


def job_environment(manifest):
    profile = manifest["profile"]
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME"):
        environment.pop(key, None)
    environment.update({key.upper(): str(profile[key]) for key in
                        ("image_path", "data_path", "output_path", "cache_path", "wandb_mode", "wandb_project")})
    environment.update(REPO_PATH=manifest["snapshot"], SLURM_SUBMIT_DIR=manifest["snapshot"])
    for prefix in ("SINGULARITYENV_", "APPTAINERENV_"):
        environment[prefix + "PYTHONPATH"] = "/workspace"
        environment[prefix + "PYTHONHOME"] = ""
        environment[prefix + "WANDB_MODE"] = profile["wandb_mode"]
        environment[prefix + "WANDB_PROJECT"] = profile["wandb_project"]
    return environment


def schedule(manifest, selected):
    run = Path(manifest["run"])
    created = []
    current = {}
    for name in manifest["stages"]:
        previous = latest_attempt(run, name)
        if name in selected:
            number = len(attempts(run, name)) + 1
            directory = run / "stages" / name / f"attempt-{number:04d}"
            directory.mkdir(parents=True)
            artifact = run / "artifacts" / name / directory.name
            atomic_json(directory / "attempt.json", {"stage": name, "artifact": str(artifact)})
            current[name] = directory
        elif previous:
            current[name] = previous
    try:
        for name in selected:
            definition = manifest["stages"][name]
            directory = current[name]
            dependencies = []
            for dependency in definition["needs"]:
                job_file = current[dependency] / "job.json"
                if job_file.exists():
                    dependencies.append(read_json(job_file)["job_id"])
            if name == "gate" and manifest["external_job"]:
                dependencies.append(manifest["external_job"])
            profile = manifest["profile"]
            resources = dict(profile["cpu"] if definition["kind"] == "cpu" else profile["gpu"])
            if definition["smoke"]:
                resources.update(profile["smoke"])
            command = ["sbatch", "--parsable", "--mail-type=NONE", "--kill-on-invalid-dep=yes", "--export=ALL",
                       f"--account={profile['account']}", f"--job-name=seg-{name}",
                       f"--chdir={manifest['snapshot']}",
                       f"--output={run / 'logs' / (name + '-' + directory.name + '-%j.out')}",
                       f"--error={run / 'logs' / (name + '-' + directory.name + '-%j.err')}"]
            if set(resources) - RESOURCES:
                raise ValueError("unsupported scheduler resource keys")
            command += [f"--{key}={value}" for key, value in resources.items()]
            if dependencies:
                condition = "afterany" if name == "summary" else "afterok"
                command.append(f"--dependency={condition}:{':'.join(dependencies)}")
            command += [str(Path(manifest["snapshot"]) / "scripts/experiment.sbatch"),
                        str(run / "manifest.json"), f"{name}/{directory.name}"]
            result = execute(command, env=job_environment(manifest))
            job_id = result.stdout.strip().split(";")[0]
            if not re.fullmatch(r"[1-9][0-9]*", job_id):
                raise RuntimeError(f"sbatch returned invalid job ID: {result.stdout!r}")
            created.append(job_id)
            atomic_json(directory / "job.json", {"job_id": job_id})
        atomic_json(run / "submission.json", {"state": "submitted", "jobs": created})
    except Exception as error:
        cleanup_error = None
        if created:
            try:
                execute(["scancel", *created])
            except Exception as cancel_error:
                cleanup_error = str(cancel_error)
        atomic_json(run / "submission.json", {"state": "failed", "jobs": created,
                    "error": str(error), "cleanup_error": cleanup_error})
        for name in selected:
            atomic_json(current[name] / "submission.json", {"state": "failed"})
        raise


def status(run):
    run = Path(run).resolve()
    manifest = read_json(run / "manifest.json")
    records = {}
    for name in manifest["stages"]:
        directory = latest_attempt(run, name)
        if directory:
            record = {"attempt": directory.name, "state": "BLOCKED"}
            for filename in ("job.json", "state.json"):
                if (directory / filename).exists():
                    record.update(read_json(directory / filename))
            records[name] = record
    jobs = [record["job_id"] for record in records.values() if "job_id" in record]
    accounting = {}
    if jobs:
        result = execute(["sacct", "-X", "-n", "-P", "-j", ",".join(jobs), "--format=JobIDRaw,State,ExitCode"])
        for line in result.stdout.splitlines():
            columns = line.split("|")
            if len(columns) >= 3:
                accounting[columns[0]] = (columns[1].split()[0].rstrip("+"), columns[2])
    for record in records.values():
        worker_state = record["state"]
        if "job_id" in record:
            scheduler, exit_code = accounting.get(record["job_id"], ("UNKNOWN", None))
            record.update(scheduler_state=scheduler, exit_code=exit_code, state=scheduler)
            if scheduler == "COMPLETED" and (worker_state == "failed" or exit_code != "0:0"):
                record["state"] = "FAILED"
            elif scheduler == "UNKNOWN" and worker_state in {"completed", "failed"}:
                record["state"] = worker_state.upper()
    for name, record in records.items():
        if record["state"] == "COMPLETED":
            directory = latest_attempt(run, name)
            try:
                if not (directory / "state.json").exists() or read_json(directory / "state.json").get("state") != "completed":
                    raise ValueError("missing successful worker completion record")
                definition = manifest["stages"][name]
                if definition["kind"] == "gpu":
                    verified_artifact(stage_artifact(manifest, name), full=not definition["smoke"])
                elif name == "gate" and read_json(directory / "gate.json").get("state") != "passed":
                    raise ValueError("missing successful gate")
                elif name == "summary" and read_json(directory / "summary.json").get("failures"):
                    raise ValueError("scientific summary contains failures")
            except (ValueError, KeyError, OSError) as error:
                record.update(state="FAILED", error=str(error))
    return {"run": str(run), "stages": records}


def retry(run):
    run = Path(run).resolve()
    with run_lock(run):
        manifest = read_json(run / "manifest.json")
        state = status(run)["stages"]
        if any(record["state"] not in TERMINAL for record in state.values()):
            raise ValueError("cannot retry while any stage is live or scheduler status is unknown")
        selected = {name for name in manifest["stages"] if name not in state or state[name]["state"] != "COMPLETED"}
        while True:
            descendants = {name for name, definition in manifest["stages"].items()
                           if selected.intersection(definition["needs"])}
            if descendants <= selected:
                break
            selected.update(descendants)
        if selected:
            schedule(manifest, [name for name in manifest["stages"] if name in selected])
    return str(run)


def verified_artifact(directory, full=True):
    directory = Path(directory)
    metrics = read_json(directory / "metrics.json")
    if metrics.get("schema_version") != 2 or (full and metrics.get("run_kind") != "full"):
        raise ValueError(f"requires schema 2 full-run metrics: {directory}")
    for path, expected in ((directory / "checkpoints/best.ckpt", metrics.get("checkpoint_sha256")),
                           (directory / "config.yaml", metrics.get("config_sha256")),
                           (directory / "split.json", metrics.get("data", {}).get("split_sha256"))):
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"artifact hash mismatch: {path}")
    selection = metrics.get("selection", {})
    if selection.get("metric") != "val/miou" or selection.get("best_epoch", -1) < 0:
        raise ValueError("teacher selection must use validation mIoU with real training progress")
    for value in (metrics["validation"]["miou"], metrics["validation"]["per_class"]["small_rock"]["iou"]):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid validation score")
    if selection.get("best_value") != metrics["validation"]["miou"]:
        raise ValueError("selection metadata does not match restored validation score")
    return metrics


def validate_gate(baseline, teacher):
    baseline_metrics = verified_artifact(baseline)
    teacher_metrics = verified_artifact(teacher)
    if baseline_metrics["data"] != teacher_metrics["data"]:
        raise ValueError("baseline and teacher data/split protocols differ")
    if teacher_metrics.get("distillation"):
        raise ValueError("teacher must be a supervised full run")
    for label, score in (("mIoU", lambda metrics: metrics["validation"]["miou"]),
                         ("small-rock IoU", lambda metrics: metrics["validation"]["per_class"]["small_rock"]["iou"])):
        if score(teacher_metrics) <= score(baseline_metrics):
            raise ValueError(f"teacher validation {label} must strictly exceed baseline")
    return {"state": "passed", "baseline": {"path": str(baseline), "checkpoint_sha256": baseline_metrics["checkpoint_sha256"]},
            "teacher": {"path": str(teacher), "checkpoint_sha256": teacher_metrics["checkpoint_sha256"],
                        "config_sha256": teacher_metrics["config_sha256"],
                        "metrics_sha256": sha256(Path(teacher) / "metrics.json"),
                        "split_sha256": teacher_metrics["data"]["split_sha256"]}}


def protocol(directory):
    directory = Path(directory)
    try:
        import yaml
        with (directory / "config.yaml").open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
        return value if isinstance(value, dict) and "training" in value else None
    except (ImportError, OSError, ValueError):
        return None


def compare(paths):
    report = {"artifacts": [], "warnings": []}
    reference = None
    reference_protocol = None
    for value in paths:
        directory = Path(value)
        metrics = verified_artifact(directory, full=False)
        scores = {"validation_miou": metrics["validation"]["miou"],
                  "validation_small_rock_iou": metrics["validation"]["per_class"]["small_rock"]["iou"],
                  "test_miou": metrics["test"]["miou"],
                  "test_small_rock_iou": metrics["test"]["per_class"]["small_rock"]["iou"]}
        if "trimap" in metrics["test"]:
            scores["test_trimap_small_rock_iou"] = metrics["test"]["trimap"]["per_class"]["small_rock"]["iou"]
        settings = protocol(directory)
        if settings is None:
            report["warnings"].append(f"{directory}: training protocol unavailable/unverified; requires verified config.yaml and PyYAML")
        entry = {"path": str(directory), "model": metrics.get("model"), "run_kind": metrics.get("run_kind"),
                 "selection": metrics.get("selection"), "scores": scores, "distillation": metrics.get("distillation")}
        if reference is not None:
            entry["delta"] = {key: score - report["artifacts"][0]["scores"][key] for key, score in scores.items()
                              if key in report["artifacts"][0]["scores"]}
            if reference["data"] != metrics["data"]:
                report["warnings"].append(f"{directory}: data/split protocol differs")
            for key in ("teacher", "method", "settings"):
                if (reference.get("distillation") or {}).get(key) != (metrics.get("distillation") or {}).get(key):
                    report["warnings"].append(f"{directory}: distillation {key} differs")
            if settings is not None and reference_protocol is not None:
                for key in ("batch_size", "accumulate_grad_batches", "epochs", "learning_rate", "precision", "seed",
                            "weight_decay", "warmup_fraction", "backbone_learning_rate"):
                    if settings.get("training", {}).get(key) != reference_protocol.get("training", {}).get(key):
                        report["warnings"].append(f"{directory}: training {key} differs")
                for key in ("data", "loss", "distillation", "metrics"):
                    if settings.get(key) != reference_protocol.get(key):
                        report["warnings"].append(f"{directory}: {key} protocol differs")
        else:
            reference = metrics
            reference_protocol = settings
        if metrics.get("run_kind") != "full":
            report["warnings"].append(f"{directory}: not a full scientific run")
        report["artifacts"].append(entry)
    return report


def launch(args):
    index = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    if not re.fullmatch(r"0|[1-9][0-9]*", index) or int(index) >= len(args.configs):
        raise ValueError("invalid SLURM_ARRAY_TASK_ID for supplied config list")
    config = args.configs[int(index)]
    root = Path(os.environ.get("REPO_PATH", os.environ.get("SLURM_SUBMIT_DIR", Path(__file__).resolve().parents[1]))).resolve()
    environment = os.environ.copy()
    environment["REPO_PATH"] = str(root)
    for name, default in (("MODE", "offline"), ("PROJECT", "seg-opt"), ("GROUP", Path(config).stem)):
        key = "WANDB_" + name
        value = environment.setdefault(key, default)
        for prefix in ("SINGULARITYENV_", "APPTAINERENV_"):
            environment[prefix + key] = value
    command = ["bash", str(root / "scripts/train.sbatch"), "python", "-m", "scripts.train",
               "--config", config, "--devices", "1"]
    if args.smoke:
        command.append("--fast-dev-run")
    os.execvpe("bash", command, environment)


def stage_artifact(manifest, name):
    directory = latest_attempt(manifest["run"], name)
    return read_json(directory / "attempt.json")["artifact"]


def gpu_container(manifest, name, directory, artifact):
    from dataclasses import replace
    from src.utils.utils import load_config, resolved_config, write_resolved_config

    definition = manifest["stages"][name]
    config = load_config(Path("/workspace") / definition["config"])
    if definition.get("seed") is not None:
        config = replace(config, training=replace(config.training, seed=definition["seed"]))
    if name in {"kd", "kd-smoke"} and "gate" in manifest["stages"]:
        gate_directory = latest_attempt(container_path(manifest, manifest["run"]), "gate")
        gate = read_json(gate_directory / "gate.json")
        if gate.get("state") != "passed":
            raise ValueError("KD requires a passed validation gate")
        teacher = gate["teacher"]
        teacher_path = Path(container_path(manifest, teacher["path"]))
        verified = verified_artifact(teacher_path)
        if (any(verified[key] != teacher[key] for key in ("config_sha256", "checkpoint_sha256"))
                or verified["data"]["split_sha256"] != teacher["split_sha256"]
                or sha256(teacher_path / "metrics.json") != teacher["metrics_sha256"]):
            raise ValueError("teacher changed after validation gate")
        if config.distillation is None:
            raise ValueError("pipeline KD config requires distillation settings")
        teacher_config = load_config(teacher_path / "config.yaml")
        for candidate in (config, teacher_config):
            data = {"class_names": list(candidate.data.class_names),
                    "image_size": [candidate.data.image_height, candidate.data.image_width],
                    "mask_variant": candidate.data.mask_variant}
            if any(verified["data"].get(key) != value for key, value in data.items()):
                raise ValueError("KD/teacher config data differs from validated teacher metrics")
        if (teacher_config.distillation is not None or teacher_config.model.name != verified["model"]["name"]
                or sha256(config.data.split_manifest) != teacher["split_sha256"]):
            raise ValueError("teacher model or KD split does not match validation gate")
        teacher_config = replace(teacher_config,
                                 data=replace(teacher_config.data, dataset_root=config.data.dataset_root,
                                              split_manifest=str(teacher_path / "split.json")),
                                 training=replace(teacher_config.training, output_dir=str(teacher_path)))
        portable_teacher = directory / "teacher-config.yaml"
        write_resolved_config(teacher_config, portable_teacher)
        config = replace(config, distillation=replace(config.distillation, teacher=replace(
            config.distillation.teacher, config_path=str(portable_teacher),
            checkpoint_path=str(teacher_path / "checkpoints/best.ckpt"))))
    config = replace(config, training=replace(config.training, output_dir=str(artifact)))
    write_resolved_config(config, directory / "config.yaml")
    atomic_json(artifact.with_name(artifact.name + ".protocol.json"), resolved_config(config))
    command = [sys.executable, "-m", "scripts.train", "--config", str(directory / "config.yaml"),
               "--output-dir", str(artifact), "--devices", "1"]
    if definition["smoke"]:
        command.append("--fast-dev-run")
    subprocess.run(command, check=True)


def summary(manifest, directory):
    failures = {}
    paths = []
    if "baseline" in manifest["artifacts"]:
        paths.append(manifest["artifacts"]["baseline"])
    if "teacher" in manifest["artifacts"]:
        paths.append(manifest["artifacts"]["teacher"])
    for name, definition in manifest["stages"].items():
        if name == "summary":
            continue
        attempt = latest_attempt(manifest["run"], name)
        record = read_json(attempt / "state.json") if attempt and (attempt / "state.json").exists() else {}
        if record.get("state") != "completed":
            failures[name] = record.get("error", "failed/blocked or missing completion record")
        elif definition["kind"] == "gpu" and not definition["smoke"]:
            artifact = Path(stage_artifact(manifest, name))
            try:
                verified_artifact(artifact)
                paths.append(str(artifact))
            except (ValueError, KeyError, OSError) as error:
                failures[name] = str(error)
    report = {"failures": failures, "comparison": None}
    try:
        report["comparison"] = compare(paths)
    except (ValueError, KeyError, OSError) as error:
        failures["comparison"] = str(error)
    atomic_json(directory / "summary.json", report)
    if failures:
        raise ValueError("scientific run incomplete; see summary.json")


def worker(args):
    manifest = read_json(args.manifest)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/attempt-[0-9]{4,}", args.stage):
        raise ValueError("worker stage must identify a stage/attempt-NNNN")
    name, attempt_name = args.stage.split("/")
    definition = manifest["stages"][name]
    host_directory = Path(manifest["run"]) / "stages" / name / attempt_name
    directory = Path(container_path(manifest, host_directory)) if args.container else host_directory
    artifact = read_json(directory / "attempt.json")["artifact"]
    if args.container:
        gpu_container(manifest, name, directory, Path(container_path(manifest, artifact)))
        return
    if (directory / "state.json").exists():
        raise ValueError("attempt already started; use retry for a new attempt")
    atomic_json(directory / "state.json", {"state": "running"})
    try:
        for relative, expected in manifest["source_hashes"].items():
            if sha256(Path(manifest["snapshot"]) / relative) != expected:
                raise ValueError(f"source snapshot changed since submission: {relative}")
        if definition["kind"] == "gpu":
            if sha256(manifest["image"]["path"]) != manifest["image"]["sha256"]:
                raise ValueError("container image changed since submission")
            command = ["bash", str(Path(manifest["snapshot"]) / "scripts/train.sbatch"),
                       "python", "-m", "scripts.experiment", "worker",
                       container_path(manifest, args.manifest), args.stage, "--container"]
            subprocess.run(command, check=True, env=job_environment(manifest), cwd=manifest["snapshot"])
        elif name == "gate":
            teacher = manifest["artifacts"].get("teacher") or stage_artifact(manifest, "teacher")
            atomic_json(directory / "gate.json", validate_gate(manifest["artifacts"]["baseline"], teacher))
        elif name == "summary":
            summary(manifest, directory)
        atomic_json(directory / "state.json", {"state": "completed"})
    except Exception as error:
        atomic_json(directory / "state.json", {"state": "failed", "error": str(error)})
        raise


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command in {"submit", "pipeline"}:
            result = {"run": submit(args)}
        elif args.command == "status":
            result = status(args.run)
        elif args.command == "retry":
            result = {"run": retry(args.run)}
        elif args.command == "compare":
            result = compare(args.artifacts)
        elif args.command == "launch":
            return launch(args)
        else:
            return worker(args)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        parser_message = str(error)
        print(f"error: {parser_message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())