from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import signal
import sys
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import (
    TRANSCRIPTION_BACKENDS,
    Settings,
    load_settings,
    validate_run_settings,
)
from .doctor import advisory_failures, fatal_failures, run_doctor
from .job import YakiFlowJob
from .media import MediaSource
from .models import JobEvent
from .models_manager import fetch_model
from .interactive_agent import (
    run_interactive_agent,
    run_memory_conflict_agent,
    start_agent_file_display,
)
from .memory import MemoryDestinationConflict


def _resume_command(work_dir: Path) -> str:
    return shlex.join(("yakiflow", "resume", str(work_dir)))


def _print_preserved(job: YakiFlowJob) -> None:
    """Point at the resume command, unless there is nothing left to resume.

    A job that already published and was reviewed refuses to run again, so
    offering its resume command would only repeat the same failure.
    """
    if job.is_finished:
        return
    print(
        f"Job preserved. Resume with: {_resume_command(job.work_dir)}",
        file=sys.stderr,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yakiflow")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a new subtitle job")
    run.add_argument("input")
    _settings_arguments(run)
    resume = sub.add_parser("resume", help="resume a preserved work directory")
    resume.add_argument("workdir", type=Path)
    _config_arguments(resume)
    for parser_ in (run, resume):
        parser_.add_argument(
            "--skip-health-check",
            action="store_true",
            help=(
                "start without checking the dependencies this configuration "
                "needs; a missing one then stops the job mid-run"
            ),
        )
    doctor = sub.add_parser(
        "doctor", help="check external dependencies and authentication"
    )
    _settings_arguments(doctor)
    models = sub.add_parser("models", help="manage Whisper models")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    fetch = models_sub.add_parser("fetch")
    _config_arguments(fetch)
    fetch.add_argument("--whisper-model", type=Path)
    secret = sub.add_parser("secret", help="manage API keys in the OS keyring")
    secret_sub = secret.add_subparsers(dest="secret_command", required=True)
    for verb in ("set", "unset"):
        entry = secret_sub.add_parser(verb)
        entry.add_argument("name", choices=["elevenlabs", "fun-asr", "mai"])
    return parser


def _config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-file", type=Path)
    parser.add_argument("--profile", metavar="NAME")
    parser.add_argument(
        "-c",
        "--config",
        dest="config_items",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "override one configuration setting by its dotted TOML key, e.g. "
            "-c transcription.backend=elevenlabs; repeatable"
        ),
    )


def _settings_arguments(parser: argparse.ArgumentParser) -> None:
    _config_arguments(parser)
    parser.add_argument("-s", "--source-language")
    parser.add_argument("-t", "--target-language")
    parser.add_argument(
        "--stream", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--context-file",
        dest="context_files",
        action="append",
        type=Path,
        metavar="FILE",
        help=(
            "copy a reference file into the work directory for interactive "
            "review; repeatable"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--output-mode", choices=["source", "translated", "bilingual", "all"]
    )
    parser.add_argument(
        "--transcription-backend", choices=list(TRANSCRIPTION_BACKENDS)
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--keep-workdir", action=argparse.BooleanOptionalAction, default=None
    )


def parse_config_items(items: Sequence[str]) -> dict[str, Any]:
    """Turn repeated ``-c key=value`` options into one nested mapping.

    Values parse as TOML literals so booleans, numbers, and arrays keep their
    types; anything that does not parse is taken as a plain string, which is
    what an unquoted language code or path already is.
    """
    result: dict[str, Any] = {}
    for item in items:
        key, sep, raw_value = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(
                f"invalid -c option {item!r}; expected KEY=VALUE with a "
                "dotted TOML key"
            )
        try:
            value = tomllib.loads(f"v = {raw_value}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw_value
        node = result
        parts = key.split(".")
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value
    return result


def _flag_settings(namespace: argparse.Namespace) -> dict[str, Any]:
    """Collect the dedicated settings flags into one nested mapping."""
    values: dict[str, Any] = {}

    def put(path: tuple[str, ...], value: Any) -> None:
        if value is None:
            return
        node = values
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = value

    get = lambda name: getattr(namespace, name, None)  # noqa: E731
    put(("source_language",), get("source_language"))
    put(("target_language",), get("target_language"))
    put(("stream",), get("stream"))
    put(("context_files",), get("context_files"))
    put(("output_dir",), get("output_dir"))
    put(("output_mode",), get("output_mode"))
    put(("transcription", "backend"), get("transcription_backend"))
    put(("work_dir",), get("work_dir"))
    put(("keep_workdir",), get("keep_workdir"))
    put(("whisper", "model"), get("whisper_model"))
    return values


def _settings(namespace: argparse.Namespace):
    config_file = getattr(namespace, "config_file", None)
    if config_file is not None and not config_file.is_file():
        # An unreadable config path would otherwise fall back to the defaults
        # and silently run with a different backend, model, or output location.
        raise ValueError(f"configuration file does not exist: {config_file}")
    return load_settings(
        _flag_settings(namespace),
        project_file=config_file,
        profile=getattr(namespace, "profile", None),
        cli_config=parse_config_items(getattr(namespace, "config_items", [])),
    )


def _resume_overrides(namespace: argparse.Namespace) -> Mapping[str, Any]:
    return parse_config_items(getattr(namespace, "config_items", []))


def _health_check(settings: Settings, input_value: str, *, reuse_live: bool = False) -> None:
    """Refuse to start a run whose environment cannot carry it through.

    Every fatal failure is reported at once, so one startup tells the whole
    story instead of each stage stopping the job on its own missing piece.
    Advisory failures are only warned about and never hold up a run.
    """
    checks = run_doctor(
        settings.for_stream() if settings.stream else settings,
        source_is_url=MediaSource.parse(input_value).is_url,
        **({"transcription_required": False, "alignment_required": False} if reuse_live else {}),
    )
    for check in advisory_failures(checks):
        print(f"yakiflow: warning: {check.name}: {check.detail}", file=sys.stderr)
    failures = fatal_failures(checks)
    if failures:
        listed = "".join(
            f"\n  {check.name}: {check.detail}" for check in failures
        )
        raise RuntimeError(
            f"health check failed:{listed}\n"
            "Fix these, or pass --skip-health-check to start anyway."
        )


def _read_secret(prompt: str) -> str:
    if sys.stdin.isatty():
        import getpass

        return getpass.getpass(prompt).strip()
    return sys.stdin.readline().strip()


def _secret(namespace: argparse.Namespace) -> int:
    try:
        import keyring
        import keyring.errors
    except ImportError:
        raise ValueError(
            "the keyring library is not installed; install yakiflow[keyring] "
            "to store secrets in the OS keyring"
        ) from None
    KEYRING_SERVICE = "yakiflow"
    KEYRING_ENTRY = namespace.name

    if namespace.secret_command == "set":
        key = _read_secret(f"{namespace.name} API key: ")
        if not key:
            raise ValueError("no API key was provided")
        keyring.set_password(KEYRING_SERVICE, KEYRING_ENTRY, key)
        print(f"Stored {namespace.name} API key in the OS keyring.")
        return 0
    try:
        keyring.delete_password(KEYRING_SERVICE, KEYRING_ENTRY)
    except keyring.errors.PasswordDeleteError:
        print(f"No {namespace.name} API key was stored.", file=sys.stderr)
        return 0
    print(f"Removed {namespace.name} API key from the OS keyring.")
    return 0


async def _run_job(job: YakiFlowJob) -> int:
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(job.run())
    interrupts = 0

    def interrupt() -> None:
        nonlocal interrupts
        interrupts += 1
        if interrupts == 1:
            print("\nStopping acquisition; finalizing available media. Press Ctrl-C again to force exit.", file=sys.stderr)
            task.cancel()
        else:
            os._exit(130)

    signal_handler_installed = False
    if hasattr(loop, "add_signal_handler"):
        try:
            loop.add_signal_handler(signal.SIGINT, interrupt)
        except NotImplementedError:
            pass
        else:
            signal_handler_installed = True
    try:
        outputs = await task
    except asyncio.CancelledError:
        print(f"Interrupted. Resume with: {_resume_command(job.work_dir)}", file=sys.stderr)
        return 130
    finally:
        if signal_handler_installed and hasattr(loop, "remove_signal_handler"):
            loop.remove_signal_handler(signal.SIGINT)
    return await _complete_job(job, outputs)


async def _complete_job(
    job: YakiFlowJob,
    outputs: Sequence[Path],
) -> int:
    if sys.stdin.isatty() and sys.stdout.isatty():
        display = await start_agent_file_display(job.settings, job.work_dir, outputs)
        try:
            await run_interactive_agent(
                job.settings,
                job.work_dir,
                outputs,
                context_files=job.context_files,
                # What stopped the previous publish attempt is the first thing
                # the next review session needs to know.
                previous_problems=job.db.get_checkpoint("review_problems", []),
                runner=job.runner,
            )
        finally:
            await display.close()
    while True:
        try:
            outputs = job.finalize_artifacts()
            break
        except MemoryDestinationConflict as conflict:
            await run_memory_conflict_agent(
                job.settings,
                job.work_dir,
                conflict.destination,
                conflict.diff(),
                runner=job.runner,
            )
            job.accept_memory_destination_change(conflict)
    for path in outputs:
        print(path)
    job.finish_review()
    cleanup = (
        job.temporary_workdir
        and not job.settings.keep_workdir
    )
    work_dir = job.work_dir
    job.close(cleanup=cleanup)
    if not cleanup:
        print(f"Work directory: {work_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    job: YakiFlowJob | None = None
    try:
        if args.command == "secret":
            return _secret(args)
        if args.command == "models":
            settings = _settings(args)
            def progress(done: int, total: int | None) -> None:
                suffix = f"/{total}" if total else ""
                print(f"\rDownloading {done}{suffix} bytes", end="", file=sys.stderr)
            path = fetch_model(settings.whisper.model, progress)
            print(f"\n{path}")
            return 0
        if args.command == "doctor":
            settings = _settings(args)
            checks = run_doctor(settings)
            for check in checks:
                # WARN is what a run only warns about; FAIL is what stops one.
                status = "OK" if check.ok else ("FAIL" if check.fatal else "WARN")
                print(f"{status:4} {check.name}: {check.detail}")
            return 1 if fatal_failures(checks) else 0
        if args.command == "resume":
            job = YakiFlowJob.from_workdir(
                args.workdir,
                listener=_print_event,
                config_file=args.config_file,
                profile=args.profile,
                overrides=_resume_overrides(args),
            )
            # The stored configuration bypasses argparse, so re-check it here
            # instead of failing deep inside a resumed stage.
            reuse_live = bool(job.settings.stream and job.db.artifact("reference_audio"))
            validate_run_settings(job.settings, check_credentials=not reuse_live)
            if not args.skip_health_check:
                _health_check(job.settings, job.input_value, reuse_live=reuse_live)
        else:
            settings = _settings(args)
            validate_run_settings(settings)
            if not args.skip_health_check:
                # Before the work directory exists: a run that cannot finish
                # should leave nothing behind to resume or clean up.
                _health_check(settings, args.input)
            job = YakiFlowJob(args.input, settings, listener=_print_event)
        job.skip_health_check = args.skip_health_check
        if sys.stdin.isatty() and sys.stdout.isatty():
            from . import tui

            succeeded = tui.run(job)
            if not succeeded:
                _print_preserved(job)
                return 0 if getattr(job, "stream_exit_requested", False) else 2
            return asyncio.run(_complete_job(job, job.outputs))
        return asyncio.run(_run_job(job))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"yakiflow: {exc}", file=sys.stderr)
        if job is not None:
            _print_preserved(job)
        return 2


async def _print_event(event: JobEvent) -> None:
    if event.kind in {"stage", "warning", "failed", "published"}:
        print(f"[{event.kind}] {event.message}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
