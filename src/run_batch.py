"""Sequential single-GPU queue on an immutable prepared dataset."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys

import yaml

try:
    from .config_utils import PROJECT_ROOT, get_required_path, load_config, resolve_project_path
    from .train_rfdetr import preflight_check
    from .training_report import check_identity, read_json, run_identity, write_csv, write_json
except ImportError:
    from config_utils import PROJECT_ROOT, get_required_path, load_config, resolve_project_path
    from train_rfdetr import preflight_check
    from training_report import check_identity, read_json, run_identity, write_csv, write_json


def task_configs(manifest):
    base = load_config(resolve_project_path(manifest["base_config"]))
    names = set()
    result = []
    for task in manifest["tasks"]:
        name = str(task["name"])
        if name in names or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Invalid/duplicate task name: {name}")
        names.add(name)
        config = copy.deepcopy(base)
        config["training"].update(task.get("training", {}))
        result.append((name, config))
    if not result:
        raise ValueError("Queue must contain tasks")
    return result


def summarize(queue_dir, state):
    rows = []
    for name, task in state["tasks"].items():
        row = {"task": name, "status": task["status"], "experiment": task.get("experiment"),
               "error": task.get("error")}
        if task["status"] == "success":
            report = read_json(Path(task["experiment"]) / "results" / "training_report.json")
            row.update(completed_epochs=report["completed_epochs"], best_epoch=report["best_epoch"],
                       train_loss=report["last_epoch_metrics"].get("train/loss"),
                       val_loss=report["last_epoch_metrics"].get("val/loss"),
                       data_sha256=report["identity"]["data_sha256"],
                       **report["evaluation"]["metrics"])
        rows.append(row)
    write_json(queue_dir / "summary.json", rows)
    write_csv(queue_dir / "summary.csv", rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "configs" / "batch_v2.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-failed", action="store_true", help="Recover incomplete tasks into a new attempt directory")
    args = parser.parse_args(argv)
    manifest = load_config(args.manifest)
    tasks = task_configs(manifest)
    planned = []
    for name, config in tasks:
        preflight_check(config, args.manifest)
        planned.append((name, config, run_identity(config)))
    if len({identity["data_sha256"] for _, _, identity in planned}) != 1:
        raise ValueError("All tasks must share the exact same prepared dataset")
    queue_dir = resolve_project_path(manifest["queue_dir"])
    state_path = queue_dir / "queue_status.json"
    state = read_json(state_path) if state_path.exists() else {"tasks": {}}
    if state["tasks"] and set(state["tasks"]) != {name for name, _, _ in planned}:
        raise ValueError("Queue task list changed; choose a new queue_dir")
    for name, config, identity in planned:
        if name in state["tasks"]:
            check_identity(identity, state["tasks"][name]["identity"])
        else:
            state["tasks"][name] = {"status": "pending", "identity": identity, "attempts": []}
        print(f"{name}: epochs={config['training']['epochs']}, lr={config['training']['learning_rate']}, "
              f"accum={config['training']['grad_accum_steps']}, data={identity['data_sha256']}")
    if args.dry_run:
        print("Dry run passed. No directories created; no model constructed.")
        return 0
    queue_dir.mkdir(parents=True, exist_ok=True)
    # Windows releases the advisory lock even when the queue process crashes.
    import msvcrt
    lock = (queue_dir / "queue.lock").open("a+b")
    lock.write(b"0")
    lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock.close()
        raise RuntimeError("This queue is already running")
    try:
        # Another invocation may have completed while this one was validating.
        if state_path.exists():
            latest = read_json(state_path)
            if set(latest["tasks"]) != {name for name, _, _ in planned}:
                raise ValueError("Queue task list changed while acquiring the lock")
            for name, _, identity in planned:
                check_identity(latest["tasks"][name]["identity"], identity)
            state = latest
        write_json(state_path, state)
        for name, config, identity in planned:
            task = state["tasks"][name]
            if task["status"] == "success":
                report = read_json(Path(task["experiment"]) / "results" / "training_report.json")
                check_identity(report["identity"], identity)
                print(f"Skipping successful task: {name}")
                continue
            previous = Path(task["experiment"]) if task.get("experiment") else None
            if previous and not args.resume_failed:
                print(f"Incomplete task {name}; use --resume-failed to recover")
                continue
            try:
                check_identity(run_identity(config), identity)
                previous_state = read_json(previous / "run_status.json") if previous and (previous / "run_status.json").exists() else {}
                if previous_state.get("training_completed"):
                    experiment = previous
                    command_args = ["--report-only", str(previous)]
                else:
                    config_path = queue_dir / f"{name}.yaml"
                    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
                    attempt_name = f"{queue_dir.name}-{name}-{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                    experiment = get_required_path(config, "experiment", "root") / attempt_name
                    command_args = ["--config", str(config_path), "--experiment-name", attempt_name]
                    if previous and (previous / "checkpoints" / "last.ckpt").is_file():
                        command_args += ["--resume-from", str(previous)]
                task.update(status="running", experiment=str(experiment))
                task["attempts"].append(str(experiment))
                write_json(state_path, state)
                log_path = queue_dir / f"{name}-{len(task['attempts'])}.log"
                print(f"Running {name}; log: {log_path}", flush=True)
                with log_path.open("w", encoding="utf-8") as stream:
                    result = subprocess.run([sys.executable, str(PROJECT_ROOT / "src" / "train_rfdetr.py"), *command_args],
                                            cwd=PROJECT_ROOT, env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
                                            stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"Exit {result.returncode}; see {log_path}")
                report = read_json(experiment / "results" / "training_report.json")
                check_identity(report["identity"], identity)
                task.update(status="success", error=None)
            except Exception as exc:
                task.update(status="failed", error=str(exc))
                print(f"Failed {name}: {exc}")
            write_json(state_path, state)
            summarize(queue_dir, state)
        summarize(queue_dir, state)
        print(f"Batch summary: {queue_dir / 'summary.csv'}")
        return 0 if all(t["status"] == "success" for t in state["tasks"].values()) else 1
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()


if __name__ == "__main__":
    sys.exit(main())