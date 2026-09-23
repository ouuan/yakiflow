"""Private subprocess entry point for WhisperX alignment and CUDA probing."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

from .alignment import (
    AlignmentModelDecisionRequired,
    _WhisperXInProcessBackend,
    _cue_from_json,
    _cue_to_json,
)


async def _worker_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Worker-local replacement for to_thread.

    The worker is already isolated from the TUI; keeping its model operations
    in one thread also avoids inheriting a parent executor during startup.
    """
    return func(*args, **kwargs)


def _send(event: dict[str, Any]) -> None:
    sys.__stdout__.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


async def _align(request: dict[str, Any]) -> None:
    asyncio.to_thread = _worker_to_thread  # type: ignore[assignment]
    cues = [_cue_from_json(item) for item in request.get("cues", [])]
    backend = _WhisperXInProcessBackend(
        language=request.get("language"),
        device=str(request.get("device", "auto")),
        model_name=request.get("model_name"),
        fallback_on_error=False,
    )

    async def progress(completed: int, total: int) -> None:
        _send({"kind": "progress", "completed": completed, "total": total})

    async def warning(message: str) -> None:
        _send({"kind": "warning", "message": message})

    # Third-party imports occasionally print diagnostics. Keep stdout a strict
    # JSON-lines protocol while preserving those diagnostics for the parent.
    with contextlib.redirect_stdout(sys.stderr):
        try:
            result = await backend.align(
                Path(str(request["audio"])),
                cues,
                vad_intervals=[tuple(item) for item in request.get("vad_intervals", [])],
                on_warning=warning,
                on_progress=progress,
            )
        except AlignmentModelDecisionRequired as exc:
            _send({"kind": "error", "category": "model_failure", "message": str(exc)})
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _send({"kind": "error", "category": "fallback", "message": f"{type(exc).__name__}: {exc}"})
            return
    _send({
        "kind": "result",
        "backend": result.backend,
        "warning": result.warning,
        "low_confidence_ids": result.low_confidence_ids,
        "cues": [_cue_to_json(cue) for cue in result.cues],
    })


def _probe() -> None:
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import torch

            available = bool(torch.cuda.is_available())
        _send({"kind": "probe", "cuda_available": available,
               "detail": "CUDA available" if available else "CUDA unavailable"})
    except Exception as exc:
        _send({"kind": "probe", "cuda_available": False,
               "detail": str(exc), "category": type(exc).__name__})


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.probe:
        _probe()
        return 0
    raw = sys.stdin.readline()
    try:
        request = json.loads(raw)
        if not isinstance(request, dict):
            raise ValueError("request must be an object")
    except Exception as exc:
        _send({"kind": "error", "category": "worker", "message": f"invalid request: {exc}"})
        return 2
    asyncio.run(_align(request))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
