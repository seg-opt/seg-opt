import concurrent.futures
from contextlib import redirect_stderr
from dataclasses import asdict, dataclass
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from scripts import experiment


REAL_RUN = subprocess.run


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repo"
        self.root.mkdir()
        for directory in ("src", "scripts/clusters", "experiments", "tests/unit"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        actual = Path(experiment.__file__).resolve().parent
        for name in ("experiment.py", "experiment.sbatch", "train.sbatch"):
            (self.root / "scripts" / name).write_bytes((actual / name).read_bytes())
        for name in ("student", "teacher", "kd"):
            (self.root / "experiments" / (name + ".yaml")).write_text("model: fixture\n")
        (self.root / "src/model.py").write_text("VALUE = 1\n")
        (self.root / ".env.secret").write_text("SECRET=not-for-snapshot")
        (self.root / "src/.env.secret").write_text("SECRET=not-for-snapshot")
        (self.root / "src/__pycache__").mkdir()
        (self.root / "src/__pycache__/cached.pyc").write_bytes(b"cached")
        self.output = Path(self.temporary.name) / "output"
        self.output.mkdir()
        self.image = Path(self.temporary.name) / "current.sif"
        self.image.write_bytes(b"image fixture")
        self.profile = self.root / "scripts/clusters/test.json"
        self.profile.write_text(json.dumps({
            "account": "fixture", "image_path": str(self.image),
            "data_path": str(self.root), "output_path": str(self.output),
            "cache_path": str(self.output / "cache"),
            "gpu": {"partition": "proxima", "constraint": "h100", "gpus-per-node": 1,
                    "cpus-per-task": 8, "mem": "128G", "time": "3-00:00:00"},
            "smoke": {"mem": "64G", "time": "00:30:00"},
            "cpu": {"partition": "proxima-cpu", "cpus-per-task": 1,
                    "mem": "1G", "time": "00:10:00"},
        }))
        self.environment = patch.dict(os.environ, {"REPO_PATH": str(self.root)}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.calls = []
        self.states = {}
        self.fail_submission = None
        self.scheduler = patch.object(experiment.subprocess, "run", side_effect=self.fake_scheduler)
        self.scheduler.start()
        self.addCleanup(self.scheduler.stop)

    def fake_scheduler(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if command[0] == "sbatch":
            number = sum(call[0][0] == "sbatch" for call in self.calls)
            if number == self.fail_submission:
                raise subprocess.CalledProcessError(1, command, stderr="fake rejected job")
            return subprocess.CompletedProcess(command, 0, str(100 + number) + ";cluster\n", "")
        if command[0] == "sacct":
            text = "\n".join(f"{job}|{state}|{exit_code}" for job, (state, exit_code) in self.states.items())
            return subprocess.CompletedProcess(command, 0, text, "")
        if command[0] == "scancel":
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(f"unexpected executable: {command}")

    def submit(self, *arguments):
        args = experiment.build_parser().parse_args([*arguments, "--profile", str(self.profile)])
        return Path(experiment.submit(args))

    def manifest(self, run):
        return json.loads((run / "manifest.json").read_text())

    def artifact(self, name, miou=0.8, small_rock=0.6, settings=None):
        directory = self.output / name
        (directory / "checkpoints").mkdir(parents=True)
        (directory / "checkpoints/best.ckpt").write_bytes(name.encode())
        (directory / "split.json").write_text('{"train":["a"],"val":["b"],"test":["c"]}')
        (directory / "config.yaml").write_text(json.dumps(settings) if settings is not None else "model: fixture\n")
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        metrics = {
            "schema_version": 2, "run_kind": "full", "model": {"name": name},
            "data": {"class_names": ["background", "sky", "small_rock", "large_rock"],
                     "image_size": [512, 768], "mask_variant": "clean",
                     "counts": {"train": 1, "val": 1, "test": 1},
                     "split_sha256": sha(directory / "split.json")},
            "selection": {"metric": "val/miou", "best_epoch": 1, "best_value": miou},
            "checkpoint_sha256": sha(directory / "checkpoints/best.ckpt"),
            "config_sha256": sha(directory / "config.yaml"),
            "validation": {"miou": miou, "per_class": {"small_rock": {"iou": small_rock}}},
            "test": {"miou": 0.9, "per_class": {"small_rock": {"iou": 0.7}}},
        }
        (directory / "metrics.json").write_text(json.dumps(metrics))
        return directory

    def pipeline(self, teacher=None):
        baseline = self.artifact("baseline")
        arguments = ["pipeline", "--config", "experiments/kd.yaml",
                     "--baseline-artifacts", str(baseline)]
        if teacher is None:
            arguments += ["--teacher-config", "experiments/teacher.yaml"]
        else:
            arguments += ["--teacher-artifacts", str(teacher), "--after-job", "99"]
        return self.submit(*arguments)

    def complete_stage(self, run, name):
        directory = experiment.latest_attempt(run, name)
        experiment.atomic_json(directory / "state.json", {"state": "completed"})
        if name == "summary":
            experiment.atomic_json(directory / "summary.json", {"failures": {}})
        elif name == "gate":
            experiment.atomic_json(directory / "gate.json", {"state": "passed"})
        else:
            source = self.artifact("completed-" + name)
            target = Path(experiment.stage_artifact(self.manifest(run), name))
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            if name.endswith("-smoke"):
                metrics = json.loads((target / "metrics.json").read_text())
                metrics["run_kind"] = "smoke"
                (target / "metrics.json").write_text(json.dumps(metrics))

    def test_shared_parser_exposes_submission_and_pipeline(self):
        parser = experiment.build_parser()
        arguments = parser.parse_args(["submit", "experiments/example.yaml"])
        self.assertEqual(arguments.command, "submit")
        self.assertEqual(arguments.configs, ["experiments/example.yaml"])

    def test_submit_seed_is_persisted_for_every_gpu_stage_without_config_changes(self):
        for seed in (0, 42, 43, 44, 4294967295):
            with self.subTest(seed=seed):
                run = self.submit("submit", "experiments/student.yaml", "experiments/teacher.yaml",
                                  "--seed", str(seed))
                manifest = self.manifest(run)
                for name in ("student-smoke", "student", "teacher-smoke", "teacher"):
                    self.assertEqual(manifest["stages"][name]["seed"], seed)
                self.assertNotIn("seed", manifest["stages"]["summary"])
                for name in ("student", "teacher"):
                    relative = f"experiments/{name}.yaml"
                    self.assertEqual((self.root / relative).read_text(), "model: fixture\n")
                    self.assertEqual((run / "source" / relative).read_bytes(),
                                     (self.root / relative).read_bytes())

    def test_pipeline_seed_is_persisted_only_for_gpu_stages_with_either_teacher_mode(self):
        teacher = self.artifact("fixed-teacher", 0.85, 0.7)
        baseline = self.output / "planned-baseline"
        for teacher_arguments, expected_gpu in (
                (["--teacher-config", "experiments/teacher.yaml"],
                 ("teacher-smoke", "teacher", "kd-smoke", "kd")),
                (["--teacher-artifacts", str(teacher), "--after-job", "99"], ("kd-smoke", "kd"))):
            with self.subTest(teacher_arguments=teacher_arguments):
                run = self.submit("pipeline", "--config", "experiments/kd.yaml",
                                  "--baseline-artifacts", str(baseline), *teacher_arguments,
                                  "--seed", "43")
                manifest = self.manifest(run)
                for name in expected_gpu:
                    self.assertEqual(manifest["stages"][name]["seed"], 43)
                self.assertNotIn("seed", manifest["stages"]["summary"])
                self.assertEqual(manifest["stages"]["gate"], {
                    "config": None, "needs": ["teacher"] if "teacher" in expected_gpu else [],
                    "kind": "cpu", "smoke": False})
                self.assertEqual(manifest["artifacts"]["baseline"], str(baseline))
                if "teacher" not in expected_gpu:
                    self.assertEqual(manifest["artifacts"]["teacher"], str(teacher))
                    gate_job = experiment.read_json(run / "stages/gate/attempt-0001/job.json")
                    gate_command = next(command for command, _ in self.calls
                                        if command[0] == "sbatch" and
                                        command[-1] == "gate/attempt-0001" and
                                        str(run / "manifest.json") in command)
                    self.assertIn("--dependency=afterok:99", gate_command)
                    self.assertTrue(gate_job["job_id"])
                self.assertFalse(baseline.exists())
                self.assertEqual((teacher / "config.yaml").read_text(), "model: fixture\n")
                self.assertEqual((self.root / "experiments/kd.yaml").read_text(), "model: fixture\n")

    def test_invalid_cli_seeds_are_rejected_before_snapshot_or_scheduler_submission(self):
        for arguments in (["submit", "experiments/student.yaml"],
                          ["pipeline", "--config", "experiments/kd.yaml",
                           "--baseline-artifacts", str(self.output / "baseline"),
                           "--teacher-artifacts", str(self.output / "teacher")]):
            for seed in ("-1", "4294967296", "1.5", "True", "False", "seed", "nan"):
                with self.subTest(command=arguments[0], seed=seed):
                    errors = StringIO()
                    with patch.object(experiment, "snapshot") as snapshot, redirect_stderr(errors), \
                            self.assertRaises(SystemExit) as stopped:
                        experiment.main([*arguments, "--seed", seed, "--profile", str(self.profile)])
                    self.assertEqual(stopped.exception.code, 2)
                    self.assertIn("seed", errors.getvalue())
                    snapshot.assert_not_called()
                    self.assertEqual(self.calls, [])
                    self.assertFalse((self.output / "experiments").exists())

    def test_no_seed_override_keeps_legacy_stage_definitions(self):
        for run in (self.submit("submit", "experiments/student.yaml"), self.pipeline()):
            with self.subTest(run=run):
                self.assertTrue(all("seed" not in definition
                                    for definition in self.manifest(run)["stages"].values()))

    def test_submit_snapshots_source_and_queues_only_snapshot_entrypoints(self):
        run = self.submit("submit", str(self.root / "experiments/student.yaml"))
        manifest = self.manifest(run)
        snapshot = run / "source"
        self.assertEqual((snapshot / "src/model.py").read_text(), "VALUE = 1\n")
        self.assertFalse((snapshot / "src/.env.secret").exists())
        self.assertFalse((snapshot / "src/__pycache__").exists())
        self.assertEqual(manifest["image"]["sha256"], hashlib.sha256(b"image fixture").hexdigest())
        self.assertEqual(manifest["profile"]["image_path"], str(self.image.resolve()))
        self.assertIn("scripts/experiment.py", manifest["source_hashes"])
        jobs = [call for call in self.calls if call[0][0] == "sbatch"]
        self.assertEqual(len(jobs), 3)
        for command, kwargs in jobs:
            self.assertIn(str(snapshot / "scripts/experiment.sbatch"), command)
            self.assertIn("--mail-type=NONE", command)
            self.assertIn("--kill-on-invalid-dep=yes", command)
            self.assertEqual(kwargs["env"]["REPO_PATH"], str(snapshot))
            self.assertEqual(kwargs["env"]["SINGULARITYENV_WANDB_MODE"], "offline")
            self.assertEqual(kwargs["env"]["APPTAINERENV_WANDB_MODE"], "offline")
        self.assertIn("--dependency=afterok:101", jobs[1][0])
        self.assertIn("--dependency=afterany:101:102", jobs[2][0])
        self.assertNotIn("SECRET", (run / "manifest.json").read_text())
        self.assertFalse((run / "artifacts/student/attempt-0001").exists())
        second = self.submit("submit", "experiments/student.yaml")
        self.assertNotEqual(run, second)

    def test_pipeline_submits_entire_dag_with_cpu_gate_and_afterany_summary(self):
        run = self.pipeline()
        commands = [call[0] for call in self.calls if call[0][0] == "sbatch"]
        self.assertEqual(list(self.manifest(run)["stages"]),
                         ["teacher-smoke", "teacher", "gate", "kd-smoke", "kd", "summary"])
        self.assertEqual(len(commands), 6)
        for index in range(1, 5):
            self.assertIn(f"--dependency=afterok:{100 + index}", commands[index])
        self.assertIn("--partition=proxima-cpu", commands[2])
        self.assertFalse(any("gpus" in option for option in commands[2]))
        self.assertIn("--mem=128G", commands[1])
        self.assertIn("--time=00:30:00", commands[0])
        self.assertIn("--dependency=afterany:101:102:103:104:105", commands[5])

    def test_partial_submission_cancels_only_jobs_created_in_call(self):
        self.fail_submission = 3
        with self.assertRaises((RuntimeError, subprocess.CalledProcessError)):
            self.pipeline()
        cancellations = [call[0] for call in self.calls if call[0][0] == "scancel"]
        self.assertEqual(cancellations, [["scancel", "101", "102"]])
        records = list((self.output / "experiments").glob("*/submission.json"))
        self.assertEqual(json.loads(records[0].read_text())["state"], "failed")

    def test_unsafe_configs_rejected_before_submission(self):
        outside = Path(self.temporary.name) / "outside.yaml"
        outside.write_text("secret")
        (self.root / "experiments/escape.yaml").symlink_to(outside)
        for name in (str(outside), "experiments/escape.yaml", "experiments/missing.yaml"):
            with self.subTest(name=name), self.assertRaises((ValueError, FileNotFoundError)):
                self.submit("submit", name)
        self.assertEqual(self.calls, [])

    def test_snapshot_rejects_escaping_symlinks(self):
        (self.root / "src/leak").symlink_to(Path(self.temporary.name))
        with self.assertRaises(ValueError):
            self.submit("submit", "experiments/student.yaml")
        self.assertEqual(self.calls, [])

    def test_snapshot_rejects_symlink_to_non_allowlisted_repository_content(self):
        (self.root / "private.txt").write_text("private")
        (self.root / "src/leak").symlink_to(self.root / "private.txt")
        with self.assertRaises(ValueError):
            self.submit("submit", "experiments/student.yaml")
        self.assertEqual(self.calls, [])

    def test_adoption_maps_external_results_and_rejects_unbound_paths(self):
        teacher = self.artifact("adopted", 0.85, 0.7)
        run = self.pipeline(teacher)
        command = self.calls[0][0]
        self.assertIn("--dependency=afterok:99", command)
        self.assertEqual(experiment.container_path(self.manifest(run), str(teacher)), "/output/adopted")
        args = experiment.build_parser().parse_args([
            "pipeline", "--config", "experiments/kd.yaml", "--baseline-artifacts", str(self.root),
            "--teacher-artifacts", str(teacher), "--profile", str(self.profile)])
        with self.assertRaisesRegex(ValueError, "OUTPUT_PATH"):
            experiment.submit(args)

    def test_gate_requires_full_matching_verified_artifacts_and_strict_validation_gains(self):
        baseline = self.artifact("baseline")
        teacher = self.artifact("teacher", 0.85, 0.7)
        result = experiment.validate_gate(baseline, teacher)
        self.assertEqual(result["teacher"]["checkpoint_sha256"],
                         hashlib.sha256(b"teacher").hexdigest())
        metrics_path = teacher / "metrics.json"
        original = json.loads(metrics_path.read_text())
        for change in ("smoke", "split", "tie", "nan", "config", "checkpoint", "selection"):
            metrics = json.loads(json.dumps(original))
            if change == "smoke":
                metrics["run_kind"] = "smoke"
            elif change == "split":
                metrics["data"]["split_sha256"] = "wrong"
            elif change == "tie":
                metrics["validation"]["per_class"]["small_rock"]["iou"] = 0.6
            elif change == "nan":
                metrics["validation"]["miou"] = float("nan")
            elif change == "selection":
                metrics["selection"]["metric"] = "test/miou"
            else:
                metrics[change + "_sha256"] = "wrong"
            metrics_path.write_text(json.dumps(metrics))
            with self.subTest(change=change), self.assertRaises(ValueError):
                experiment.validate_gate(baseline, teacher)

    def test_status_uses_scheduler_and_retry_rejects_live_stages(self):
        run = self.submit("submit", "experiments/student.yaml")
        self.states = {"101": ("COMPLETED", "0:0"), "102": ("RUNNING", "0:0"),
                       "103": ("PENDING", "0:0")}
        status = experiment.status(run)
        self.assertEqual(status["stages"]["student"]["state"], "RUNNING")
        with self.assertRaisesRegex(ValueError, "live"):
            experiment.retry(run)
        self.states = {}
        with self.assertRaisesRegex(ValueError, "unknown"):
            experiment.retry(run)
        self.assertEqual(sum(call[0][0] == "sbatch" for call in self.calls), 3)

    def test_retry_preserves_completed_attempts_and_uses_new_failed_attempts(self):
        run = self.submit("submit", "experiments/student.yaml")
        self.complete_stage(run, "student-smoke")
        self.complete_stage(run, "summary")
        self.states = {"101": ("COMPLETED", "0:0"), "102": ("FAILED", "1:0"),
                       "103": ("COMPLETED", "0:0")}
        experiment.retry(run)
        status = experiment.status(run)
        self.assertEqual(status["stages"]["student-smoke"]["attempt"], "attempt-0001")
        self.assertEqual(status["stages"]["student"]["attempt"], "attempt-0002")
        self.assertEqual(status["stages"]["summary"]["attempt"], "attempt-0002")
        self.assertFalse((run / "artifacts/student/attempt-0002").exists())

    def test_retry_missing_teacher_renews_completed_descendants_in_manifest_order(self):
        run = self.pipeline()
        manifest = self.manifest(run)
        for name in manifest["stages"]:
            self.complete_stage(run, name)
        self.states = {str(job): ("COMPLETED", "0:0") for job in range(101, 107)}
        checkpoint = Path(experiment.stage_artifact(manifest, "teacher")) / "checkpoints/best.ckpt"
        checkpoint.unlink()
        self.assertEqual(experiment.status(run)["stages"]["teacher"]["state"], "FAILED")
        old_files = {path: path.read_bytes() for parent in (run / "stages", run / "artifacts")
                     for path in parent.rglob("*") if path.is_file()}
        experiment.retry(run)
        self.assertEqual(experiment.latest_attempt(run, "teacher-smoke").name, "attempt-0001")
        for name in ("teacher", "gate", "kd-smoke", "kd", "summary"):
            with self.subTest(stage=name):
                self.assertEqual(experiment.latest_attempt(run, name).name, "attempt-0002")
                self.assertEqual(experiment.stage_artifact(manifest, name),
                                 str(run / "artifacts" / name / "attempt-0002"))
                self.assertFalse(Path(experiment.stage_artifact(manifest, name)).exists())
        jobs = [command for command, _ in self.calls if command[0] == "sbatch"][6:]
        self.assertEqual([command[-1] for command in jobs],
                         [name + "/attempt-0002" for name in ("teacher", "gate", "kd-smoke", "kd", "summary")])
        self.assertIn("--dependency=afterok:107", jobs[1])
        self.assertIn("--dependency=afterok:109", jobs[3])
        self.assertIn("--dependency=afterany:101:107:108:109:110", jobs[4])
        self.assertTrue(all(path.read_bytes() == content for path, content in old_files.items()))

    def test_retry_missing_kd_keeps_valid_teacher_gate_and_smoke(self):
        run = self.pipeline()
        manifest = self.manifest(run)
        for name in manifest["stages"]:
            self.complete_stage(run, name)
        self.states = {str(job): ("COMPLETED", "0:0") for job in range(101, 107)}
        (Path(experiment.stage_artifact(manifest, "kd")) / "checkpoints/best.ckpt").unlink()
        experiment.retry(run)
        for name in ("teacher-smoke", "teacher", "gate", "kd-smoke"):
            self.assertEqual(experiment.latest_attempt(run, name).name, "attempt-0001")
        jobs = [command for command, _ in self.calls if command[0] == "sbatch"][6:]
        self.assertEqual([command[-1] for command in jobs], ["kd/attempt-0002", "summary/attempt-0002"])
        self.assertIn("--dependency=afterok:104", jobs[0])

    def test_concurrent_atomic_stage_records_do_not_lose_updates(self):
        run = self.submit("submit", "experiments/student.yaml")
        manifest_before = (run / "manifest.json").read_bytes()
        paths = [run / "stages" / name / "attempt-0001/state.json"
                 for name in ("student-smoke", "student", "summary")]
        with concurrent.futures.ThreadPoolExecutor() as pool:
            list(pool.map(lambda path: experiment.atomic_json(path, {"state": "completed"}), paths))
        self.assertTrue(all(json.loads(path.read_text())["state"] == "completed" for path in paths))
        self.assertEqual((run / "manifest.json").read_bytes(), manifest_before)

    def test_compare_rejects_changed_or_deleted_artifact_files(self):
        baseline = self.artifact("baseline")
        for filename in ("config.yaml", "split.json", "checkpoints/best.ckpt"):
            for mutation in ("changed", "deleted"):
                with self.subTest(file=filename, mutation=mutation):
                    artifact = self.artifact(filename.replace("/", "-") + "-" + mutation)
                    path = artifact / filename
                    if mutation == "changed":
                        path.write_bytes(b"tampered")
                    else:
                        path.unlink()
                    with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                        experiment.compare([baseline, artifact])

    def test_compare_downgrades_unbound_sidecars_even_with_valid_config_hash(self):
        baseline = self.artifact("baseline")
        for location in ("neighbor", "parent", "artifact"):
            artifact = self.artifact("unbound-" + location)
            path = {"neighbor": artifact.with_name(artifact.name + ".protocol.json"),
                    "parent": artifact.parent / "protocol.json",
                    "artifact": artifact / "protocol.json"}[location]
            path.write_text(json.dumps({"config_sha256": experiment.sha256(artifact / "config.yaml"),
                                        "training": {"batch_size": 999}}))
            with self.subTest(location=location), patch.dict("sys.modules", {"yaml": None}):
                report = experiment.compare([baseline, artifact])
                self.assertTrue(any(str(artifact) in warning and "unverified" in warning
                                    for warning in report["warnings"]))
            path.unlink()

    def test_compare_verified_yaml_overrides_tampered_sidecar(self):
        baseline = self.artifact("baseline")
        kd = self.artifact("kd")
        for directory in (baseline, kd):
            config = directory / "config.yaml"
            config.write_text(json.dumps({"training": {"batch_size": 16}}))
            metrics = experiment.read_json(directory / "metrics.json")
            metrics["config_sha256"] = experiment.sha256(config)
            experiment.atomic_json(directory / "metrics.json", metrics)
        sidecar = kd.with_name(kd.name + ".protocol.json")
        sidecar.write_text(json.dumps({"config_sha256": experiment.sha256(kd / "config.yaml"),
                                       "training": {"batch_size": 999}}))
        parser = ModuleType("yaml")
        parser.safe_load = json.load
        with patch.dict("sys.modules", {"yaml": parser}):
            report = experiment.compare([baseline, kd])
        self.assertEqual(report["warnings"], [])
        metrics = experiment.read_json(kd / "metrics.json")
        metrics["run_kind"] = "smoke"
        experiment.atomic_json(kd / "metrics.json", metrics)
        with patch.dict("sys.modules", {"yaml": parser}):
            report = experiment.compare([baseline, kd])
        self.assertEqual(len(report["artifacts"]), 2)
        self.assertTrue(any("not a full scientific run" in warning for warning in report["warnings"]))

    def test_compare_reports_deltas_and_protocol_warnings_without_ranking(self):
        baseline = self.artifact("baseline")
        kd = self.artifact("kd", 0.82, 0.65)
        for directory, batch in ((baseline, 16), (kd, 8)):
            (directory / "config.yaml").write_text(json.dumps({
                "training": {"batch_size": batch, "accumulate_grad_batches": 16 // batch},
                "distillation": None if batch == 16 else {"method": "vanilla"}}))
            metrics = experiment.read_json(directory / "metrics.json")
            metrics["config_sha256"] = experiment.sha256(directory / "config.yaml")
            if directory == kd:
                metrics["distillation"] = {"method": "vanilla", "settings": {"weight": 1}}
            experiment.atomic_json(directory / "metrics.json", metrics)
        parser = ModuleType("yaml")
        parser.safe_load = json.load
        with patch.dict("sys.modules", {"yaml": parser}):
            report = experiment.compare([baseline, kd])
        self.assertAlmostEqual(report["artifacts"][1]["delta"]["validation_miou"], 0.02)
        self.assertTrue(any("batch" in warning for warning in report["warnings"]))
        self.assertTrue(any("distillation method differs" in warning for warning in report["warnings"]))
        self.assertNotIn("ranking", report)
        with patch.dict("sys.modules", {"yaml": None}):
            report = experiment.compare([baseline, kd])
        self.assertTrue(any("training protocol unavailable" in warning for warning in report["warnings"]))

    def test_compare_does_not_trust_runner_protocol_without_yaml(self):
        baseline = self.artifact("baseline")
        kd = self.artifact("kd", 0.82, 0.65)
        for directory, batch in ((baseline, 16), (kd, 8)):
            directory.with_name(directory.name + ".protocol.json").write_text(json.dumps({
                "training": {"batch_size": batch, "accumulate_grad_batches": 16 // batch}}))
        with patch.dict("sys.modules", {"yaml": None}):
            report = experiment.compare([baseline, kd])
        self.assertEqual(sum("training protocol unavailable/unverified" in warning
                             for warning in report["warnings"]), 2)
        self.assertFalse(any("batch_size" in warning for warning in report["warnings"]))

    def test_compare_warns_on_each_optimizer_protocol_difference(self):
        training = {"batch_size": 16, "accumulate_grad_batches": 1, "epochs": 20,
                    "learning_rate": 0.0005, "precision": "bf16", "seed": 42,
                    "weight_decay": 0.05, "warmup_fraction": 0.05,
                    "backbone_learning_rate": 0.00003, "output_dir": "/output/baseline"}
        baseline = self.artifact("baseline", settings={"training": training})
        parser = ModuleType("yaml")
        parser.safe_load = json.load
        for field, value in (("weight_decay", 0.1), ("warmup_fraction", 0.1),
                             ("backbone_learning_rate", 0.00006)):
            artifact = self.artifact(field, settings={"training": {**training, field: value}})
            with self.subTest(field=field), patch.dict("sys.modules", {"yaml": parser}):
                report = experiment.compare([baseline, artifact])
                self.assertEqual(report["warnings"], [f"{artifact}: training {field} differs"])
        routed = self.artifact("routed", settings={"training": {**training, "output_dir": "/output/kd"}})
        with patch.dict("sys.modules", {"yaml": parser}):
            report = experiment.compare([baseline, routed])
        self.assertEqual(report["warnings"], [])

    def test_compare_uses_real_verified_yaml_when_parser_is_available(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("real PyYAML comparison is checked in the CPU container")
        training = {"batch_size": 16, "weight_decay": 0.05, "warmup_fraction": 0.05,
                    "backbone_learning_rate": 0.00003, "output_dir": "/output/baseline"}
        baseline = self.artifact("baseline")
        kd = self.artifact("kd")
        for artifact in (baseline, kd):
            (artifact / "config.yaml").write_text(yaml.safe_dump({"training": training}))
            metrics = experiment.read_json(artifact / "metrics.json")
            metrics["config_sha256"] = experiment.sha256(artifact / "config.yaml")
            experiment.atomic_json(artifact / "metrics.json", metrics)
        sidecar = kd.with_name(kd.name + ".protocol.json")
        sidecar.write_text(json.dumps({"config_sha256": experiment.sha256(kd / "config.yaml"),
                                       "training": {**training, "weight_decay": 99}}))
        self.assertEqual(experiment.compare([baseline, kd])["warnings"], [])
        for field in ("weight_decay", "warmup_fraction", "backbone_learning_rate"):
            with self.subTest(field=field):
                (kd / "config.yaml").write_text(yaml.safe_dump({"training": {**training, field: 0.1}}))
                metrics = experiment.read_json(kd / "metrics.json")
                metrics["config_sha256"] = experiment.sha256(kd / "config.yaml")
                experiment.atomic_json(kd / "metrics.json", metrics)
                self.assertEqual(experiment.compare([baseline, kd])["warnings"],
                                 [f"{kd}: training {field} differs"])

    def test_completed_scheduler_without_valid_artifacts_is_not_scientific_success(self):
        run = self.submit("submit", "experiments/student.yaml")
        self.states = {"101": ("COMPLETED", "0:0"), "102": ("COMPLETED", "0:0"),
                       "103": ("COMPLETED", "0:0")}
        record = experiment.status(run)["stages"]["student"]
        self.assertEqual(record["state"], "FAILED")

    def test_real_snapshot_shell_entrypoint_runs_gate_without_site_packages(self):
        teacher = self.artifact("teacher", 0.85, 0.7)
        run = self.pipeline(teacher)
        spooled = Path(self.temporary.name) / "slurm_script"
        spooled.write_bytes((run / "source/scripts/experiment.sbatch").read_bytes())
        (self.root / "scripts/experiment.py").write_text("raise RuntimeError('live source must not run')\n")
        result = REAL_RUN(["bash", str(spooled),
                           str(run / "manifest.json"), "gate/attempt-0001"],
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        directory = run / "stages/gate/attempt-0001"
        self.assertEqual(json.loads((directory / "state.json").read_text())["state"], "completed")
        self.assertEqual(json.loads((directory / "gate.json").read_text())["state"], "passed")
        result = REAL_RUN(["bash", str(spooled),
                           str(run / "manifest.json"), "summary/attempt-0001"],
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), text=True, capture_output=True)
        self.assertEqual(result.returncode, 1)
        report = json.loads((run / "stages/summary/attempt-0001/summary.json").read_text())
        self.assertIn("kd", report["failures"])

    def test_worker_rejects_modified_snapshot_before_dispatch(self):
        run = self.submit("submit", "experiments/student.yaml")
        (run / "source/src/model.py").write_text("CHANGED = True\n")
        args = experiment.build_parser().parse_args(["worker", str(run / "manifest.json"),
                                                    "student-smoke/attempt-0001"])
        with self.assertRaisesRegex(ValueError, "snapshot"):
            experiment.worker(args)
        state = json.loads((run / "stages/student-smoke/attempt-0001/state.json").read_text())
        self.assertEqual(state["state"], "failed")

    def test_worker_dispatch_uses_importable_module_and_bound_snapshot(self):
        run = self.submit("submit", "experiments/student.yaml")
        args = experiment.build_parser().parse_args(["worker", str(run / "manifest.json"),
                                                    "student-smoke/attempt-0001"])
        with patch.object(experiment.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as invoke:
            experiment.worker(args)
        command = invoke.call_args.args[0]
        self.assertEqual(command[:5], ["bash", str(run / "source/scripts/train.sbatch"),
                                     "python", "-m", "scripts.experiment"])
        self.assertIn("/output/experiments/" + run.name + "/manifest.json", command)
        self.assertEqual(invoke.call_args.kwargs["env"]["REPO_PATH"], str(run / "source"))

    def test_managed_schedule_and_worker_environment_reject_inherited_python_paths(self):
        poison = {"PYTHONPATH": "/outside/package", "PYTHONHOME": "/outside/python"}
        for prefix in ("SINGULARITYENV_", "APPTAINERENV_"):
            poison[prefix + "PYTHONPATH"] = "/outside/package"
            poison[prefix + "PYTHONHOME"] = "/outside/python"
        with patch.dict(os.environ, poison):
            run = self.submit("submit", "experiments/student.yaml")
            environments = [kwargs["env"] for command, kwargs in self.calls if command[0] == "sbatch"]
            args = experiment.build_parser().parse_args([
                "worker", str(run / "manifest.json"), "student-smoke/attempt-0001"])
            with patch.object(experiment.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as invoke:
                experiment.worker(args)
            environments.append(invoke.call_args.kwargs["env"])
        for environment in environments:
            with self.subTest(environment=environment["REPO_PATH"]):
                self.assertNotIn("PYTHONPATH", environment)
                self.assertNotIn("PYTHONHOME", environment)
                for prefix in ("SINGULARITYENV_", "APPTAINERENV_"):
                    self.assertEqual(environment[prefix + "PYTHONPATH"], "/workspace")
                    self.assertEqual(environment[prefix + "PYTHONHOME"], "")

    def test_profile_does_not_capture_arbitrary_secrets(self):
        profile = json.loads(self.profile.read_text())
        profile["api_token"] = "do-not-record"
        self.profile.write_text(json.dumps(profile))
        with self.assertRaisesRegex(ValueError, "profile"):
            self.submit("submit", "experiments/student.yaml")
        self.assertFalse((self.output / "experiments").exists())

    def test_profile_rejects_unsupported_resource_fields_before_snapshot(self):
        profile = json.loads(self.profile.read_text())
        profile["gpu"]["api_token"] = "do-not-record"
        self.profile.write_text(json.dumps(profile))
        with self.assertRaisesRegex(ValueError, "resource"):
            self.submit("submit", "experiments/student.yaml")
        self.assertFalse((self.output / "experiments").exists())

    def test_gpu_config_override_uses_verified_teacher_and_never_prepopulates_artifacts(self):
        @dataclass(frozen=True)
        class Training:
            output_dir: str
            seed: int = 42

        @dataclass(frozen=True)
        class Teacher:
            config_path: str
            checkpoint_path: str

        @dataclass(frozen=True)
        class Distillation:
            teacher: Teacher

        @dataclass(frozen=True)
        class Data:
            dataset_root: str
            split_manifest: str
            class_names: tuple = ("background", "sky", "small_rock", "large_rock")
            image_height: int = 512
            image_width: int = 768
            mask_variant: str = "clean"

        @dataclass(frozen=True)
        class Model:
            name: str

        @dataclass(frozen=True)
        class Config:
            training: Training
            distillation: Distillation
            data: Data
            model: Model

        teacher = self.artifact("teacher", 0.85, 0.7)
        run = self.pipeline(teacher)
        manifest = self.manifest(run)
        gate = experiment.validate_gate(self.output / "baseline", teacher)
        experiment.atomic_json(run / "stages/gate/attempt-0001/gate.json", gate)
        directory = run / "stages/kd/attempt-0001"
        artifact = Path(experiment.stage_artifact(manifest, "kd"))
        config = Config(Training("old-output"), Distillation(Teacher("old-config", "old-checkpoint")),
                Data("/datasets/artificial_lunar_landscape", str(teacher / "split.json")), Model("student"))
        teacher_config = Config(Training("/unbound/old/results"), None,
                    Data("/unbound/old/data", "/unbound/old/split.json"), Model("teacher"))
        module = ModuleType("src.utils.utils")
        module.load_config = lambda path: teacher_config if Path(path) == teacher / "config.yaml" else config
        module.resolved_config = asdict
        written = []

        def save_config(value, path):
            self.assertFalse(artifact.exists())
            written.append(value)
            Path(path).write_text(json.dumps(asdict(value)))

        module.write_resolved_config = save_config
        with patch.dict("sys.modules", {"src.utils.utils": module}), \
                patch.object(experiment, "container_path", side_effect=lambda manifest, path: str(path)), \
                patch.object(experiment.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as invoke:
            experiment.gpu_container(manifest, "kd", directory, artifact)
        rewritten = written[-1]
        self.assertEqual(rewritten.training.output_dir, str(artifact))
        self.assertEqual(rewritten.training.seed, 42)
        self.assertEqual(rewritten.data, config.data)
        self.assertEqual(rewritten.distillation.teacher.config_path, str(directory / "teacher-config.yaml"))
        self.assertEqual(rewritten.distillation.teacher.checkpoint_path, str(teacher / "checkpoints/best.ckpt"))
        saved_teacher = json.loads((directory / "teacher-config.yaml").read_text())
        self.assertEqual(saved_teacher["data"]["split_manifest"], str(teacher / "split.json"))
        self.assertEqual(saved_teacher["data"]["dataset_root"], "/datasets/artificial_lunar_landscape")
        self.assertEqual((teacher / "config.yaml").read_text(), "model: fixture\n")
        self.assertFalse(artifact.exists())
        self.assertTrue(artifact.with_name(artifact.name + ".protocol.json").is_file())
        command = invoke.call_args.args[0]
        self.assertEqual(command[1:4], ["-m", "scripts.train", "--config"])
        self.assertIn(str(directory / "config.yaml"), command)

        for name, seed in (("kd-smoke", 0), ("kd", 43)):
            with self.subTest(stage=name, seed=seed):
                manifest["stages"][name]["seed"] = seed
                directory = run / "stages" / name / "attempt-0001"
                artifact = Path(experiment.stage_artifact(manifest, name))
                with patch.dict("sys.modules", {"src.utils.utils": module}), \
                        patch.object(experiment, "container_path", side_effect=lambda manifest, path: str(path)), \
                        patch.object(experiment.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
                    experiment.gpu_container(manifest, name, directory, artifact)
                self.assertEqual(written[-1].training.seed, seed)
                self.assertEqual(written[-1].data, config.data)
                self.assertEqual(config.training.seed, 42)
                saved_teacher = json.loads((directory / "teacher-config.yaml").read_text())
                self.assertEqual(saved_teacher["training"]["seed"], 42)
                self.assertEqual((teacher / "config.yaml").read_text(), "model: fixture\n")

    def test_gpu_seed_override_resolves_baseline_config_without_changing_data_or_split(self):
        @dataclass(frozen=True)
        class Training:
            output_dir: str = "old-output"
            seed: int = 42

        @dataclass(frozen=True)
        class Data:
            dataset_root: str = "/datasets/artificial_lunar_landscape"
            split_manifest: str = "experiments/split_seed42.json"

        @dataclass(frozen=True)
        class Config:
            training: Training
            data: Data

        run = self.submit("submit", "experiments/student.yaml")
        manifest = self.manifest(run)
        config = Config(Training(), Data())
        module = ModuleType("src.utils.utils")
        module.load_config = lambda path: config
        module.resolved_config = asdict
        module.write_resolved_config = lambda value, path: Path(path).write_text(json.dumps(asdict(value)))
        for name in ("student-smoke", "student"):
            for seed in ("missing", None, 0, 43, 4294967295):
                with self.subTest(stage=name, seed=seed):
                    definition = manifest["stages"][name]
                    if seed != "missing":
                        definition["seed"] = seed
                    else:
                        definition.pop("seed", None)
                    directory = run / "stages" / name / "attempt-0001"
                    artifact = Path(experiment.stage_artifact(manifest, name))
                    with patch.dict("sys.modules", {"src.utils.utils": module}), \
                            patch.object(experiment.subprocess, "run",
                                         return_value=subprocess.CompletedProcess([], 0)) as invoke:
                        experiment.gpu_container(manifest, name, directory, artifact)
                    expected_seed = 42 if seed in ("missing", None) else seed
                    resolved = json.loads((directory / "config.yaml").read_text())
                    self.assertEqual(resolved["training"], {"output_dir": str(artifact), "seed": expected_seed})
                    self.assertEqual(resolved["data"], asdict(config.data))
                    protocol = experiment.read_json(artifact.with_name(artifact.name + ".protocol.json"))
                    self.assertEqual(protocol, resolved)
                    self.assertEqual(config.training.seed, 42)
                    self.assertEqual("--fast-dev-run" in invoke.call_args.args[0], name == "student-smoke")
                    self.assertFalse(artifact.exists())

    def test_default_profile_mounts_dataset_parent_for_existing_config_paths(self):
        profile = json.loads((Path(experiment.__file__).parent / "clusters/eagle.json").read_text())
        self.assertTrue(profile["data_path"].endswith("/datasets"))

    def test_launch_validates_index_and_executes_existing_launcher_without_config_io(self):
        arguments = experiment.build_parser().parse_args(["launch", "--smoke", "absent/a.yaml", "absent/b.yaml"])
        for index in ("-1", "01", "x", "2"):
            with patch.dict(os.environ, {"SLURM_ARRAY_TASK_ID": index}), self.assertRaises(ValueError):
                experiment.launch(arguments)
        with patch.dict(os.environ, {"SLURM_ARRAY_TASK_ID": "1"}), patch.object(experiment.os, "execvpe") as execute:
            experiment.launch(arguments)
        command = execute.call_args.args[1]
        self.assertEqual(command, ["bash", str(self.root / "scripts/train.sbatch"), "python", "-m",
                                   "scripts.train", "--config", "absent/b.yaml", "--devices", "1", "--fast-dev-run"])
        self.assertEqual(execute.call_args.args[2]["WANDB_GROUP"], "b")
        self.assertEqual(execute.call_args.args[2]["WANDB_MODE"], "offline")


if __name__ == "__main__":
    unittest.main()