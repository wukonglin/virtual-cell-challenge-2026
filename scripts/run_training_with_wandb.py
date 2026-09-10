"""Track a training command without changing its frozen model/training code."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from wandb_training import (TrainingTracker, child_environment, config_from_command,
                            global_rank, line_metrics, scalar_metrics, sha256)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("pretrain", "posttrain", "validation"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "virtual-cell-challenge-2026"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--group", default="lingshu-public")
    parser.add_argument("--name")
    parser.add_argument("--mode", choices=("offline", "online"), default="offline")
    parser.add_argument("--summary-file", type=Path, action="append", default=[])
    parser.add_argument("--parent-checkpoint-sha256")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Supply a training command after --")
    env = child_environment()
    if global_rank() != 0:
        # Non-primary ranks still execute their training task, without logging.
        return subprocess.call(command, env=env)
    os.umask(0o077)
    for path in args.summary_file:
        if path.exists():
            parser.error("Summary paths must not exist before training; stale summaries are forbidden")
    config = config_from_command(command)
    for arg in command[:2]:
        if arg.endswith(".py") and Path(arg).is_file():
            config["entrypoint_sha256"] = sha256(arg)
    if args.parent_checkpoint_sha256:
        import re
        if not re.fullmatch(r"[a-f0-9]{64}", args.parent_checkpoint_sha256):
            parser.error("Parent checkpoint SHA-256 must be 64 lowercase hex characters")
        config["parent_checkpoint_sha256"] = args.parent_checkpoint_sha256
    try:
        tracker = TrainingTracker(args.output_dir, stage=args.stage, project=args.project,
            entity=args.entity, group=args.group, name=args.name, mode=args.mode, config=config)
    except Exception as exc:
        # Do not echo exception text, which could contain credentials/SDK settings.
        print(f"Tracking initialization failed ({type(exc).__name__}); training was not started. Check tracking requirements and online configuration.", file=sys.stderr)
        return 78
    child = None
    interrupted = False
    received_signal = None
    previous = {}

    def forward(signum, frame):
        nonlocal interrupted, received_signal
        interrupted = True
        received_signal = signum
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    code = 1
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, forward)
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, errors="replace", bufsize=1, env=env, start_new_session=True)
        if received_signal is not None:
            forward(received_signal, None)
        oversized = False
        while True:
            line = child.stdout.readline(65537)
            if not line:
                break
            with contextlib.suppress(BrokenPipeError):
                sys.stdout.write(line)
                sys.stdout.flush()
            if oversized or len(line) > 65536:
                oversized = not line.endswith("\n")
                continue
            tracker.log(line_metrics(line))
        code = child.wait()
        if code < 0:
            code = 128 - code
        for path in args.summary_file:
            if path.is_file() and path.stat().st_size <= 32 * 1024 * 1024:
                try:
                    tracker.summary(json.loads(path.read_text()))
                except (ValueError, OSError):
                    tracker.failures += 1
            else:
                tracker.failures += 1
    except Exception as exc:
        print(f"Tracking wrapper error ({type(exc).__name__}); see the child output for training status.", file=sys.stderr)
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        try:
            tracker.finish(code, interrupted=interrupted)
        except Exception:
            tracker.record["tracking_errors"] = tracker.record.get("tracking_errors", 0) + 1
            print("Tracking finalization incomplete; preserving the training exit code.", file=sys.stderr)
    print(json.dumps({"tracking_mode": args.mode, "training_exit_code": code,
                      "tracking_events": tracker.events, "tracking_errors": tracker.record["tracking_errors"]}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
