import asyncio
import json
import os
import stat
import wave
from pathlib import Path

import pytest

import yakiflow.job as job_module
from conftest import make_settings
from yakiflow.alignment import AlignmentResult
from yakiflow.job import YakiFlowJob
from yakiflow.media import MediaArtifact, MediaSource
from yakiflow.memory import MemoryDestinationConflict
from yakiflow.models import AgentTraceEvent, Cue, JobEvent, TranscriptEvent
from yakiflow.process import ProcessResult
from yakiflow.subtitles import (
    ASS_HEADER,
    DEFAULT_PLAY_RES,
    AssEvent,
    ass_header,
    render_ass,
    render_events,
)
from yakiflow.translation import AgentBackend


class RecordingAwaitable:
    def __init__(self, values: list, value: object) -> None:
        self.values = values
        self.value = value

    def __await__(self):
        self.values.append(self.value)
        if False:
            yield
        return None


def ffprobe_result(args, resolution: tuple[int, int] | None) -> ProcessResult:
    """One ffprobe answer; ``None`` is an input without a picture."""
    streams = (
        []
        if resolution is None
        else [{
            "width": resolution[0],
            "height": resolution[1],
            "disposition": {"attached_pic": 0},
        }]
    )
    return ProcessResult(tuple(args), 0, json.dumps({"streams": streams}), "")


class PipelineRunner:
    def __init__(self, resolution: tuple[int, int] | None = (1920, 1080)):
        self.calls: list[list[str]] = []
        self.resolution = resolution

    async def run(self, args, *, on_line=None, **kwargs):
        args = [str(value) for value in args]
        self.calls.append(args)
        if args[0] == "ffprobe":
            return ffprobe_result(args, self.resolution)
        if args[0] == "ffmpeg":
            Path(args[-1]).write_bytes(b"RIFF-fake")
            return ProcessResult(tuple(args), 0, "", "")
        if args[0] == "whisper-cli":
            prefix = Path(args[args.index("--output-file") + 1])
            prefix.with_suffix(".json").write_text(json.dumps({
                "transcription": [
                    {"timestamps": {"from": "00:00:00,200", "to": "00:00:01,500"}, "text": "hello"},
                    {"timestamps": {"from": "00:00:01,500", "to": "00:00:02,800"}, "text": "world"},
                ]
            }))
            if on_line:
                for percent in (5, 50, 100):
                    value = on_line(
                        "stderr",
                        f"whisper_print_progress_callback: progress = {percent:3d}%",
                    )
                    if asyncio.iscoroutine(value):
                        await value
                for line in (
                    "[00:00:00.200 --> 00:00:01.500] hello",
                    "[00:00:01.500 --> 00:00:02.800] world",
                ):
                    value = on_line("stdout", line)
                    if asyncio.iscoroutine(value):
                        await value
            return ProcessResult(tuple(args), 0, "", "")
        raise AssertionError(args)


class PipelineBackend(AgentBackend):
    name = "fake"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def invoke_with_trace(
        self, prompt, *, system="", model, effort, schema, on_event=None
    ):
        self.prompts.append(prompt)
        payload = json.loads(prompt.split("INPUT:\n", 1)[1])
        return {
            "cues": [
                {"id": cue["id"], "source": cue["source"], "translated": cue.get("translated") or f"T:{cue['source']}"}
                for cue in payload["cues"]
            ],
        }


def test_context_files_are_snapshotted_and_restored_on_resume(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "chat.xml"
    second = second_dir / "chat.xml"
    first.write_text("first version", encoding="utf-8")
    second.write_text("second version", encoding="utf-8")
    work_dir = tmp_path / "work"
    job = YakiFlowJob(
        "input.mp4",
        make_settings(
            agent={"backend": "codex"},
            context_files=(first, second),
            work_dir=work_dir,
            memory=tmp_path / "memory.md",
        ),
        backend=PipelineBackend(),
    )

    assert job.context_files == (
        work_dir / "context" / "chat.xml",
        work_dir / "context" / "chat-2.xml",
    )
    assert [path.read_text(encoding="utf-8") for path in job.context_files] == [
        "first version",
        "second version",
    ]
    first.write_text("changed after job creation", encoding="utf-8")
    job.close()

    resumed = YakiFlowJob.from_workdir(work_dir, backend=PipelineBackend())
    assert resumed.context_files == (
        work_dir / "context" / "chat-2.xml",
        work_dir / "context" / "chat.xml",
    )
    assert {
        path.name: path.read_text(encoding="utf-8")
        for path in resumed.context_files
    } == {"chat.xml": "first version", "chat-2.xml": "second version"}
    resumed.close()


def test_forced_alignment_replaces_timeline_then_translation_only_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = []

    async def listener(event) -> None:
        events.append(event)

    class SourceChangingBackend(AgentBackend):
        name = "source-changing"

        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.systems: list[str] = []

        async def invoke_with_trace(
        self, prompt, *, system="", model, effort, schema, on_event=None
    ):
            self.prompts.append(prompt)
            self.systems.append(system)
            payload = json.loads(prompt.split("INPUT:\n", 1)[1])
            return {
                "cues": [
                    {
                        "id": cue["id"],
                        "source": "must be ignored",
                        "translated": f"T:{cue['source']}",
                    }
                    for cue in payload["cues"]
                ]
            }

    class SplitAligner:
        def __init__(self) -> None:
            self.calls = 0

        async def align(self, audio, cues, **kwargs):
            self.calls += 1
            await kwargs["on_progress"](1, 2)
            await kwargs["on_progress"](2, 2)
            return AlignmentResult(
                [
                    Cue("1", 0.1, 0.8, "first sentence", None, metadata={"parent_id": "old"}),
                    Cue("2", 0.8, 1.5, "second sentence", None, metadata={"parent_id": "old"}),
                ],
                "whisperx",
            )

    work_dir = tmp_path / "work"
    backend = SourceChangingBackend()
    aligner = SplitAligner()
    monkeypatch.setattr(job_module, "make_alignment_backend", lambda *args, **kwargs: aligner)
    job = YakiFlowJob(
        "input.mp4",
        make_settings(
            source_language="en",
            target_language="zh-CN",
            agent={"backend": "codex"},
            whisper={"model": tmp_path / "model.bin"},
            work_dir=work_dir,
        ),
        backend=backend,
        listener=listener,
    )
    job.db.upsert_cues([Cue("old", 0, 2, "parent", "父译文")])
    artifact = MediaArtifact(MediaSource.parse("input.mp4"), tmp_path / "audio.wav", None, True)

    aligned = asyncio.run(job._alignment_stage(artifact, job.db.list_cues()))
    translated = asyncio.run(job._post_alignment_translation_stage(artifact, aligned))
    restored = asyncio.run(job._alignment_stage(artifact, translated))

    assert aligner.calls == 1
    assert job.db.get_checkpoint("alignment_complete") is True
    assert [cue.id for cue in restored] == ["1", "2"]
    assert [cue.source for cue in restored] == ["first sentence", "second sentence"]
    assert [cue.translated for cue in restored] == [
        "T:first sentence",
        "T:second sentence",
    ]
    assert "do not modify, correct, merge, or split" in backend.systems[0]
    assert "timeline-replaced" in [event.kind for event in events]
    assert [
        event.message
        for event in events
        if event.kind == "progress" and event.stage == "align"
    ] == [
        "forced alignment · 1/2 cues",
        "forced alignment · 2/2 cues",
        "alignment complete",
    ]
    job.close()


@pytest.mark.parametrize(
    ("configured_backend", "result_backend"),
    [("whisperx", "whisperx"), ("vad", "whisper-vad"), ("vad", "unaligned")],
)
def test_alignment_stage_always_applies_volume_start_refinement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_backend: str,
    result_backend: str,
) -> None:
    class PassThroughAligner:
        async def align(self, _audio, cues, **_kwargs):
            return AlignmentResult(list(cues), result_backend)

    refine_calls = 0

    async def refine(_self, _audio, cues, **_kwargs):
        nonlocal refine_calls
        refine_calls += 1
        return [cue.with_timing(cue.start + 0.25, cue.end) for cue in cues]

    monkeypatch.setattr(
        job_module,
        "make_alignment_backend",
        lambda *args, **kwargs: PassThroughAligner(),
    )
    monkeypatch.setattr(job_module.PcmVolumeStartRefiner, "refine", refine)
    work_dir = tmp_path / f"{configured_backend}-{result_backend}"
    job = YakiFlowJob(
        "input.mp4",
        make_settings(
            source_language="en",
            target_language="zh-CN",
            alignment={"backend": configured_backend},
            agent={"backend": "codex"},
            whisper={"model": tmp_path / "model.bin"},
            work_dir=work_dir,
        ),
        backend=PipelineBackend(),
    )
    job.db.upsert_cues([Cue("1", 0.1, 1.0, "source", "translation")])
    artifact = MediaArtifact(
        MediaSource.parse("input.mp4"), tmp_path / "audio.wav", None, True
    )

    aligned = asyncio.run(job._alignment_stage(artifact, job.db.list_cues()))

    assert refine_calls == 1
    assert aligned[0].start == pytest.approx(0.35)
    assert aligned[0].end == pytest.approx(1.5)
    job.close()


@pytest.mark.parametrize(
    ("configured", "detected", "expected"),
    [("ja", "jpn", "ja"), ("ja-JP", "en", "ja"), ("auto", "ja", "ja")],
)
def test_alignment_stage_prefers_configured_source_language(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
    detected: str,
    expected: str,
) -> None:
    received_languages = []

    class PassThroughAligner:
        async def align(self, _audio, cues, **_kwargs):
            return AlignmentResult(list(cues), "whisperx")

    def make_backend(*_args, **kwargs):
        received_languages.append(kwargs["language"])
        return PassThroughAligner()

    async def keep_starts(_self, _audio, cues, **_kwargs):
        return list(cues)

    monkeypatch.setattr(job_module, "make_alignment_backend", make_backend)
    monkeypatch.setattr(job_module.PcmVolumeStartRefiner, "refine", keep_starts)
    job = YakiFlowJob(
        "input.mp4",
        make_settings(
            source_language=configured,
            target_language="zh-CN",
            alignment={"backend": "whisperx"},
            agent={"backend": "codex"},
            whisper={"model": tmp_path / "model.bin"},
            work_dir=tmp_path / "alignment-language",
        ),
        backend=PipelineBackend(),
    )
    try:
        job.db.checkpoint("detected_source_language", detected)
        job.db.upsert_cues([Cue("1", 0, 1, "source", "translation")])
        artifact = MediaArtifact(
            MediaSource.parse("input.mp4"), tmp_path / "audio.wav", None, True
        )

        asyncio.run(job._alignment_stage(artifact, job.db.list_cues()))

        assert received_languages == [expected]
    finally:
        job.close()


def test_alignment_stage_adjusts_long_vad_silence_before_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received_cues: list[Cue] = []

    class CapturingAligner:
        async def align(self, _audio, cues, **kwargs):
            received_cues.extend(cues)
            assert kwargs["vad_intervals"] == [[0, 2], [20, 25]]
            return AlignmentResult(list(cues), "whisperx")

    async def keep_starts(_self, _audio, cues, **_kwargs):
        return list(cues)

    monkeypatch.setattr(
        job_module, "make_alignment_backend", lambda *args, **kwargs: CapturingAligner()
    )
    monkeypatch.setattr(job_module.PcmVolumeStartRefiner, "refine", keep_starts)
    job = YakiFlowJob(
        "input.mp4",
        make_settings(
            source_language="en",
            target_language="zh-CN",
            alignment={"backend": "whisperx"},
            agent={"backend": "codex"},
            whisper={"model": tmp_path / "model.bin"},
            work_dir=tmp_path / "pre-align-vad",
        ),
        backend=PipelineBackend(),
    )
    job.db.upsert_cues([Cue("1", 1, 25, "source", "translation")])
    job.db.checkpoint("whisper_vad_intervals", [(0, 2), (20, 25)])
    artifact = MediaArtifact(
        MediaSource.parse("input.mp4"), tmp_path / "audio.wav", None, True
    )

    aligned = asyncio.run(job._alignment_stage(artifact, job.db.list_cues()))

    assert received_cues[0].start == 17
    assert aligned[0].start == 17
    job.close()


def test_alignment_stage_extends_ends_after_final_start_refinement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class PassThroughAligner:
        async def align(self, _audio, cues, **_kwargs):
            return AlignmentResult(list(cues), "whisperx")

    async def refine(_self, _audio, cues, **_kwargs):
        return [cues[0], cues[1].with_timing(1.5, cues[1].end)]

    monkeypatch.setattr(
        job_module, "make_alignment_backend", lambda *args, **kwargs: PassThroughAligner()
    )
    monkeypatch.setattr(job_module.PcmVolumeStartRefiner, "refine", refine)
    job = YakiFlowJob(
        "input.mp4",
        make_settings(
            source_language="en",
            target_language="zh-CN",
            alignment={"backend": "whisperx"},
            agent={"backend": "codex"},
            whisper={"model": tmp_path / "model.bin"},
            work_dir=tmp_path / "final-end-extension",
        ),
        backend=PipelineBackend(),
    )
    job.db.upsert_cues(
        [
            Cue("1", 0, 1, "first", "第一句"),
            Cue("2", 1, 2, "second", "第二句"),
        ]
    )
    artifact = MediaArtifact(
        MediaSource.parse("input.mp4"), tmp_path / "audio.wav", None, True
    )

    aligned = asyncio.run(job._alignment_stage(artifact, job.db.list_cues()))

    assert aligned[0].end == pytest.approx(1.5)
    assert aligned[1].start == pytest.approx(1.5)
    job.close()


class BlockingBackend(AgentBackend):
    name = "blocking"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def invoke_with_trace(
        self, prompt, *, system="", model, effort, schema, on_event=None
    ):
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class InterruptedWhisperRunner:
    async def run(self, args, *, on_line=None, **kwargs):
        args = [str(value) for value in args]
        if args[0] == "ffprobe":
            return ffprobe_result(args, (1920, 1080))
        assert args[0] == "whisper-cli"
        assert on_line is not None
        for line in (
            "[00:00:00.200 --> 00:00:01.500] hello",
            "[00:00:01.500 --> 00:00:02.800] world",
        ):
            await on_line("stdout", line)
        raise asyncio.CancelledError


class ResumedWhisperRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def run(self, args, *, on_line=None, **kwargs):
        args = [str(value) for value in args]
        self.calls.append(args)
        if args[0] == "ffprobe":
            return ffprobe_result(args, (1920, 1080))
        if args[0] == "ffmpeg":
            Path(args[-1]).write_bytes(b"RIFF-resume")
            return ProcessResult(tuple(args), 0, "", "")
        assert args[0] == "whisper-cli"
        prefix = Path(args[args.index("--output-file") + 1])
        prefix.with_suffix(".json").write_text(json.dumps({
            "transcription": [
                {
                    "timestamps": {
                        "from": "00:00:00,200",
                        "to": "00:00:01,200",
                    },
                    "text": "again",
                },
            ]
        }))
        if on_line:
            await on_line("stdout", "[00:00:00.200 --> 00:00:01.200] again")
        return ProcessResult(tuple(args), 0, "", "")


def test_temporary_workdirs_use_a_private_per_user_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_root = tmp_path / "yakiflow"
    monkeypatch.setattr(
        job_module, "user_runtime_path", lambda *_args, **_kwargs: expected_root
    )

    work_dir = job_module._temporary_workdir()

    assert work_dir.parent == expected_root
    assert stat.S_IMODE(expected_root.stat().st_mode) == 0o700
    assert expected_root.stat().st_uid == os.getuid()


def test_temporary_workdir_does_not_require_getuid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_root = tmp_path / "windows-runtime" / "yakiflow"
    monkeypatch.setattr(
        job_module, "user_runtime_path", lambda *_args, **_kwargs: expected_root
    )
    monkeypatch.delattr(job_module.os, "getuid")

    work_dir = job_module._temporary_workdir()

    assert work_dir.parent == expected_root


def test_temporary_workdir_rejects_an_insecure_existing_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "yakiflow"
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    monkeypatch.setattr(
        job_module, "user_runtime_path", lambda *_args, **_kwargs: root
    )

    with pytest.raises(RuntimeError, match="unsafe temporary repository"):
        job_module._temporary_workdir()


@pytest.mark.parametrize("persisted_status", [True, False])
def test_resume_preserves_temporary_workdir_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    persisted_status: bool,
) -> None:
    runtime_root = tmp_path / "yakiflow"
    monkeypatch.setattr(
        job_module, "user_runtime_path", lambda *_args, **_kwargs: runtime_root
    )
    job = YakiFlowJob(
        "input.mp4",
        make_settings(agent={"backend": "codex"}),
        backend=PipelineBackend(),
    )
    work_dir = job.work_dir
    assert job.temporary_workdir
    assert job.db.get_checkpoint("temporary_workdir") is True
    if not persisted_status:
        # Exercise the runtime-path fallback for jobs created before the
        # temporary-workdir checkpoint was introduced.
        with job.db.connection:
            job.db.connection.execute(
                "DELETE FROM checkpoints WHERE name='temporary_workdir'"
            )
    job.close()

    resumed = YakiFlowJob.from_workdir(work_dir, backend=PipelineBackend())

    assert resumed.temporary_workdir
    resumed.close(cleanup=True)
    assert not work_dir.exists()


def test_resume_preserves_effective_profile_values_and_option_tokens(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "work"
    settings = make_settings(
        source_language="auto",
        target_language="zh-CN",
        agent={
            "backend": "codex",
            "draft": {
                "batch_size": 5,
                "extra_options": ("-c", "service_tier=fast"),
            },
            "final": {"extra_options": ("--search",)},
        },
        stream=True,
        commands={"yt_dlp_options": ("--cookies-from-browser", "chrome")},
        work_dir=work_dir,
    )
    job = YakiFlowJob("input.mp4", settings, backend=PipelineBackend())
    job.close()

    resumed = YakiFlowJob.from_workdir(work_dir, backend=PipelineBackend())

    assert resumed.settings.stream is True
    assert resumed.settings.agent.draft.batch_size == 5
    assert resumed.settings.agent.draft.extra_options == (
        "-c",
        "service_tier=fast",
    )
    assert resumed.settings.agent.final.extra_options == ("--search",)
    assert resumed.settings.commands.yt_dlp_options == (
        "--cookies-from-browser",
        "chrome",
    )
    resumed.close()


def test_local_input_is_canonicalized_before_journaling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_dir = tmp_path / "launch"
    launch_dir.mkdir()
    media = launch_dir / "movie.mp4"
    media.write_bytes(b"media")
    work_dir = tmp_path / "work"
    monkeypatch.chdir(launch_dir)

    job = YakiFlowJob(
        "movie.mp4",
        make_settings(
            agent={"backend": "codex"},
            memory=tmp_path / "memory.md",
            work_dir=work_dir,
        ),
        backend=PipelineBackend(),
    )
    assert job.input_value == str(media.resolve())
    assert job.db.job()["input"] == str(media.resolve())
    job.close()

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)
    resumed = YakiFlowJob.from_workdir(work_dir, backend=PipelineBackend())
    assert resumed.input_value == str(media.resolve())
    assert not resumed.temporary_workdir
    resumed.close()


def test_new_job_rejects_populated_workdir_and_resume_keeps_one_job_row(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "work"
    settings = make_settings(
        agent={"backend": "codex"},
        memory=tmp_path / "memory.md",
        work_dir=work_dir,
    )
    job = YakiFlowJob("first.mp4", settings, backend=PipelineBackend())
    assert job.db.connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    job.close()

    with pytest.raises(ValueError, match="yakiflow resume"):
        YakiFlowJob("second.mp4", settings, backend=PipelineBackend())

    for _ in range(2):
        resumed = YakiFlowJob.from_workdir(work_dir, backend=PipelineBackend())
        assert resumed.input_value == str((Path.cwd() / "first.mp4").resolve())
        assert (
            resumed.db.connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            == 1
        )
        resumed.close()

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("existing")
    with pytest.raises(ValueError, match="work directory is not empty"):
        YakiFlowJob(
            "third.mp4",
            make_settings(
                agent={"backend": "codex"},
                memory=tmp_path / "memory.md",
                work_dir=unrelated,
            ),
            backend=PipelineBackend(),
        )


def test_agent_summary_counts_real_concurrent_operations(tmp_path: Path) -> None:
    settings = make_settings(
        source_language="auto",
        target_language="zh-CN",
        agent={"backend": "codex"},
        whisper={"model": tmp_path / "model.bin"},
        memory=tmp_path / "memory.md",
        work_dir=tmp_path / "work",
    )
    events = []

    def listener(event) -> RecordingAwaitable:
        return RecordingAwaitable(events, event)

    job = YakiFlowJob(
        "input.mp4", settings, backend=PipelineBackend(), listener=listener
    )

    async def exercise() -> None:
        await job.agent_event(AgentTraceEvent(
            "one", "lifecycle", "first", state="running"
        ))
        await job.agent_event(AgentTraceEvent(
            "two", "lifecycle", "second", state="running"
        ))
        await job.agent_event(AgentTraceEvent(
            "one", "lifecycle", "first", state="completed"
        ))
        await job.agent_event(AgentTraceEvent(
            "two", "lifecycle", "second", state="completed"
        ))

    asyncio.run(exercise())
    summaries = [event.message for event in events if event.kind == "agent"]
    assert summaries[1].endswith("2 active")
    assert summaries[-1] == "Agent · Idle"
    job.close()


def test_published_subtitles_match_the_video_resolution(tmp_path: Path) -> None:
    media = tmp_path / "vertical.mp4"
    media.write_bytes(b"media")
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    work_dir = tmp_path / "work"
    settings = make_settings(
        source_language="en", target_language="zh-CN", whisper={"model": model},
        memory=tmp_path / "memory.md", output_dir=tmp_path / "out",
        work_dir=work_dir,
        agent={"backend": "codex"},
    )
    runner = PipelineRunner(resolution=(1080, 1920))
    job = YakiFlowJob(
        str(media), settings, runner=runner, backend=PipelineBackend()
    )

    outputs = asyncio.run(job.run())

    expected = ass_header((1080, 1920))
    assert outputs[0].read_text(encoding="utf-8").startswith(expected)
    # The live preview is watched against the video too, so it cannot be
    # written at a different resolution than the published file.
    partial = tmp_path / "out" / "vertical.draft.incomplete.ass"
    assert partial.read_text(encoding="utf-8").startswith(expected)
    assert job.play_res == (1080, 1920)
    job.close()

    # A resumed job republishes and canonicalizes at the same resolution
    # instead of probing again.
    resumed = YakiFlowJob.from_workdir(work_dir, backend=PipelineBackend())
    assert resumed.play_res == (1080, 1920)
    resumed.close()


def test_input_without_a_picture_keeps_the_default_resolution(
    tmp_path: Path,
) -> None:
    media = tmp_path / "podcast.m4a"
    media.write_bytes(b"media")
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    settings = make_settings(
        source_language="en", target_language="zh-CN", whisper={"model": model},
        memory=tmp_path / "memory.md", output_dir=tmp_path / "out",
        work_dir=tmp_path / "work",
        agent={"backend": "codex"},
    )
    job = YakiFlowJob(
        str(media),
        settings,
        runner=PipelineRunner(resolution=None),
        backend=PipelineBackend(),
    )

    outputs = asyncio.run(job.run())

    assert outputs[0].read_text(encoding="utf-8").startswith(ASS_HEADER)
    assert job.play_res == DEFAULT_PLAY_RES
    job.close()


def test_unavailable_ffprobe_warns_and_publishes_at_the_default_resolution(
    tmp_path: Path,
) -> None:
    media = tmp_path / "movie.mp4"
    media.write_bytes(b"media")
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    settings = make_settings(
        source_language="en", target_language="zh-CN", whisper={"model": model},
        memory=tmp_path / "memory.md", output_dir=tmp_path / "out",
        work_dir=tmp_path / "work",
        agent={"backend": "codex"},
    )

    class MissingFfprobeRunner(PipelineRunner):
        async def run(self, args, *, on_line=None, **kwargs):
            if str(args[0]) == "ffprobe":
                raise FileNotFoundError("ffprobe")
            return await super().run(args, on_line=on_line, **kwargs)

    events: list[JobEvent] = []

    async def listener(event) -> None:
        events.append(event)

    job = YakiFlowJob(
        str(media),
        settings,
        runner=MissingFfprobeRunner(),
        backend=PipelineBackend(),
        listener=listener,
    )

    outputs = asyncio.run(job.run())

    assert outputs[0].read_text(encoding="utf-8").startswith(ASS_HEADER)
    assert any(
        event.kind == "warning" and "video resolution" in event.message
        for event in events
    )
    job.close()


def test_local_job_runs_full_pipeline_with_fake_processes(tmp_path: Path) -> None:
    media = tmp_path / "movie.mp4"
    media.write_bytes(b"media")
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    settings = make_settings(
        source_language="auto", target_language="zh-CN", whisper={"model": model},
        memory=tmp_path / "memory.md", output_dir=tmp_path / "out", work_dir=tmp_path / "work",
        output_mode="all",
        agent={"backend": "codex", "draft": {"model": "draft"}, "final": {"model": "final"}},
    )
    runner = PipelineRunner()
    events = []
    transcript_snapshots: list[str] = []

    async def listener(event):
        events.append(event)
        if event.kind == "transcript":
            partial = tmp_path / "out" / "movie.draft.incomplete.ass"
            assert partial.is_file()
            transcript_snapshots.append(partial.read_text(encoding="utf-8"))

    job = YakiFlowJob(
        str(media), settings, runner=runner, backend=PipelineBackend(), listener=listener
    )
    outputs = asyncio.run(job.run())
    assert len(outputs) == 3 and all(path.exists() for path in outputs)
    bilingual = next(path for path in outputs if path.name.endswith("auto-zh-cn.ass")).read_text()
    assert bilingual.startswith(ASS_HEADER)
    assert r"T:hello\Nhello" in bilingual
    assert job.db.get_checkpoint("transcribed") is True
    whisper_calls = [call for call in runner.calls if call[0] == "whisper-cli"]
    assert len(whisper_calls) == 1
    assert whisper_calls[0][whisper_calls[0].index("-mc") + 1] == "0"
    assert "--print-progress" in whisper_calls[0]
    assert whisper_calls[0][whisper_calls[0].index("--language") + 1] == "auto"
    partial = tmp_path / "out" / "movie.draft.incomplete.ass"
    assert partial.exists()
    assert "hello" in transcript_snapshots[0]
    assert "world" in transcript_snapshots[-1]
    assert not (tmp_path / "work" / "movie.draft.incomplete.ass").exists()
    processed = [event.cue for event in events if event.kind == "subtitle"]
    assert processed and all(cue and cue.translated for cue in processed)
    event_kinds = [event.kind for event in events]
    assert "transcript" in event_kinds
    assert event_kinds.index("transcript") < event_kinds.index("subtitle")
    agent_messages = [event.message for event in events if event.kind == "agent"]
    assert any("Running" in message for message in agent_messages)
    assert agent_messages[-1] == "Agent · Idle"
    progress = [event.progress for event in events if event.progress is not None]
    assert progress == sorted(progress)
    assert progress[-1] == 1
    progress_events = [event for event in events if event.progress is not None]
    assert all(event.stage for event in progress_events)
    assert progress_events[-1].estimated_remaining == 0
    assert any("Whisper transcription · 50%" in event.message for event in events)
    assert any("Agent draft translation · 1/1 batches" in event.message for event in events)
    assert "translate" not in job._progress_plan.ranges
    assert any(
        event.stage == "transcribe" and "Agent draft translation" in event.message
        for event in events
    )
    job.finish_review()
    assert job.db.job()["status"] == "complete"
    job.close()


def test_reviewing_resume_preserves_and_finalizes_staged_agent_edits(
    tmp_path: Path,
) -> None:
    media = tmp_path / "movie.mp4"
    media.write_bytes(b"media")
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    work_dir = tmp_path / "work"
    output_dir = tmp_path / "out"
    settings = make_settings(
        source_language="auto",
        target_language="zh-CN",
        agent={"backend": "codex"},
        whisper={"model": model},
        memory=tmp_path / "memory.md",
        output_dir=output_dir,
        work_dir=work_dir,
    )
    job = YakiFlowJob(
        str(media), settings, runner=PipelineRunner(), backend=PipelineBackend()
    )
    reviewed = render_ass([Cue("1", 0, 1, "reviewed by Agent", "评审后的字幕")])
    staged = asyncio.run(job.run())[0]
    staged.write_text(reviewed, encoding="utf-8")
    job.close()

    backend = PipelineBackend()
    resumed = YakiFlowJob.from_workdir(work_dir, backend=backend)
    restored = asyncio.run(resumed.run())

    assert restored == [staged]
    assert staged.read_text(encoding="utf-8") == reviewed
    assert backend.prompts == []
    finalized = resumed.finalize_artifacts()
    assert finalized == [output_dir / staged.name]
    assert finalized[0].read_text(encoding="utf-8") == reviewed
    resumed.close()


def test_finalize_requires_all_staged_subtitles_before_moving_any(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    work_dir = tmp_path / "work"
    job = YakiFlowJob(
        "input.mp4",
        make_settings(
            agent={"backend": "codex"},
            memory=tmp_path / "memory.md",
            output_dir=output_dir,
            work_dir=work_dir,
        ),
        backend=PipelineBackend(),
    )
    present = work_dir / "movie.source.ass"
    missing = work_dir / "movie.translated.ass"
    present.write_text("new source\n", encoding="utf-8")
    output_dir.mkdir()
    stale_destination = output_dir / missing.name
    stale_destination.write_text("stale translation\n", encoding="utf-8")
    job.outputs = [present, missing]
    job._output_destination_base = output_dir / "movie"

    with pytest.raises(RuntimeError, match="missing staged subtitle files"):
        job.finalize_artifacts()

    assert present.is_file()
    assert not (output_dir / present.name).exists()
    assert stale_destination.read_text(encoding="utf-8") == "stale translation\n"
    assert job.outputs == [present, missing]
    job.close()


def _review_job(tmp_path: Path, cues: list[Cue], output_mode: str = "bilingual"):
    job = YakiFlowJob(
        "input.mp4",
        make_settings(
            source_language="en",
            target_language="zh-CN",
            agent={"backend": "codex"},
            memory=tmp_path / "memory.md",
            output_dir=tmp_path / "out",
            work_dir=tmp_path / "work",
            output_mode=output_mode,
        ),
        backend=PipelineBackend(),
    )
    job.db.upsert_cues(cues, stable=True)
    job._output_destination_base = tmp_path / "out" / "movie"
    return job


def test_finalize_rejects_a_merge_applied_to_only_one_artifact(tmp_path: Path) -> None:
    job = _review_job(
        tmp_path,
        [Cue("1", 0, 1, "one", "T:one"), Cue("2", 1, 2, "two", "T:two")],
        output_mode="all",
    )
    merged = render_events([AssEvent(0.0, 2.0, "", ("T:one T:two",))])
    split = render_events([
        AssEvent(0.0, 1.0, "", ("one",)),
        AssEvent(1.0, 2.0, "", ("two",)),
    ])
    translated = job.work_dir / "movie.translated.ass"
    source = job.work_dir / "movie.source.ass"
    translated.write_text(merged, encoding="utf-8")
    source.write_text(split, encoding="utf-8")
    job.outputs = [translated, source]

    with pytest.raises(RuntimeError, match="not publishable"):
        job.finalize_artifacts()

    assert translated.is_file() and source.is_file()
    assert not (tmp_path / "out" / translated.name).exists()
    job.close()


def test_finalize_canonicalizes_disorder_left_by_a_merge(tmp_path: Path) -> None:
    # A hand merge deleted one Dialogue line, left the events out of order,
    # edited the header, and used SRT-habit timestamps. All of that is
    # mechanically repairable: finalize rewrites the header wholesale,
    # re-sorts the events, and normalizes the timestamps.
    job = _review_job(
        tmp_path,
        [
            Cue("1", 0, 1, "one", "T:one"),
            Cue("2", 1, 2, "two", "T:two"),
            Cue("3", 2, 3, "three", "T:three"),
        ],
    )
    staged = job.work_dir / "movie.en-zh-cn.ass"
    staged.write_text(
        "[Script Info]\n"
        "Title: hand-edited during review\n"
        "ScriptType: v4.00+\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:02.00,0:00:03.00,Default,,0,0,0,,T:three\\Nthree\n"
        "Dialogue: 0,0:00:00.000,0:00:02.000,Default,,0,0,0,,T:one T:two\n",
        encoding="utf-8",
    )
    job.outputs = [staged]

    published = job.finalize_artifacts()

    assert published[0].read_text(encoding="utf-8") == render_events([
        AssEvent(0.0, 2.0, "", ("T:one T:two",)),
        AssEvent(2.0, 3.0, "", ("T:three", "three")),
    ])
    assert published[0].read_text(encoding="utf-8").startswith(ASS_HEADER)
    job.close()


def test_finalize_tolerates_timing_defects_the_pipeline_itself_produced(
    tmp_path: Path,
) -> None:
    # Whisper can emit overlapping unnamed cues; the review is not what broke
    # them.
    cues = [Cue("1", 0, 2, "one", "T:one"), Cue("2", 1, 3, "two", "T:two")]
    job = _review_job(tmp_path, cues)
    staged = job.work_dir / "movie.en-zh-cn.ass"
    staged.write_text(render_ass(cues), encoding="utf-8")
    job.outputs = [staged]

    published = job.finalize_artifacts()

    assert published == [tmp_path / "out" / staged.name]
    job.close()


def test_finalize_refuses_a_staged_file_the_review_emptied(tmp_path: Path) -> None:
    job = _review_job(
        tmp_path,
        [Cue("1", 0, 1, "one", "T:one"), Cue("2", 1, 2, "two", "T:two")],
    )
    staged = job.work_dir / "movie.en-zh-cn.ass"
    staged.write_text("", encoding="utf-8")
    job.outputs = [staged]
    destination = tmp_path / "out" / staged.name
    destination.parent.mkdir(parents=True)
    destination.write_text("published subtitles\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="contains no subtitle cues"):
        job.finalize_artifacts()

    assert destination.read_text(encoding="utf-8") == "published subtitles\n"
    job.close()


def test_validation_failure_checkpoints_review_problems_until_repaired(
    tmp_path: Path,
) -> None:
    cues = [Cue("1", 0, 1, "one", "T:one"), Cue("2", 1, 2, "two", "T:two")]
    job = _review_job(tmp_path, cues)
    staged = job.work_dir / "movie.en-zh-cn.ass"
    staged.write_text("", encoding="utf-8")
    job.outputs = [staged]

    with pytest.raises(RuntimeError, match="not publishable"):
        job.finalize_artifacts()

    problems = job.db.get_checkpoint("review_problems")
    assert isinstance(problems, list) and problems
    assert all(staged.name in problem for problem in problems)

    staged.write_text(render_ass(cues), encoding="utf-8")
    job.finalize_artifacts()

    assert job.db.get_checkpoint("review_problems") == []
    job.close()


def test_finalize_refuses_a_staged_file_saved_in_another_encoding(
    tmp_path: Path,
) -> None:
    job = _review_job(tmp_path, [Cue("1", 0, 1, "one", "T:字幕")])
    staged = job.work_dir / "movie.en-zh-cn.ass"
    original = render_ass([Cue("1", 0, 1, "one", "T:字幕")]).encode("gbk")
    staged.write_bytes(original)
    job.outputs = [staged]

    with pytest.raises(RuntimeError, match="not valid UTF-8"):
        job.finalize_artifacts()

    # Reading leniently and writing the result back would have replaced the
    # reviewed text with U+FFFD before anyone could re-save it.
    assert staged.read_bytes() == original
    job.close()


def test_finalize_tolerates_artifact_drift_the_pipeline_itself_produced(
    tmp_path: Path,
) -> None:
    # A cue Whisper left without source text renders as an empty Dialogue,
    # which only the source-language artifact drops. The review did not cause
    # that.
    cues = [Cue("1", 0, 1, "one", "T:one"), Cue("2", 1, 2, "", "T:two")]
    job = _review_job(tmp_path, cues, output_mode="all")
    names = ("movie.source.ass", "movie.translated.ass", "movie.en-zh-cn.ass")
    staged = [job.work_dir / name for name in names]
    staged[0].write_text(
        render_events([AssEvent(0.0, 1.0, "", ("one",))]), encoding="utf-8"
    )
    staged[1].write_text(render_ass(cues, "translated"), encoding="utf-8")
    staged[2].write_text(render_ass(cues, "bilingual"), encoding="utf-8")
    job.outputs = staged

    published = job.finalize_artifacts()

    assert published == [tmp_path / "out" / name for name in names]
    job.close()


# An ASS file another tool wrote: its own header, styles, comment lines,
# override tags, layers, and margins, none of which YakiFlow would produce.
AEGISUB_ASS = (
    "\ufeff[Script Info]\n"
    "; Script generated by Aegisub 3.2.2\n"
    "Title: Episode 1\n"
    "ScriptType: v4.00+\n"
    "PlayResX: 1280\n"
    "PlayResY: 720\n"
    "\n"
    "[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
    "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
    "MarginR, MarginV, Encoding\n"
    "Style: Default,Noto Sans CJK SC,48,&H00FFFFFF,&H000000FF,&H00000000,"
    "&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,20,20,20,1\n"
    "Style: Sign,Noto Serif CJK SC,40,&H0000FFFF,&H000000FF,&H00000000,"
    "&H00000000,1,0,0,0,100,100,0,0,1,2,0,8,20,20,20,1\n"
    "\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    "Comment: 0,0:00:00.00,0:00:05.00,Default,,0,0,0,,translator note\n"
    "Dialogue: 0,0:00:01.00,0:00:02.50,Default,Aki,0,0,0,,{\\i1}你好{\\i0}\\Nこんにちは\n"
    "Dialogue: 1,0:00:03.00,0:00:04.00,Sign,,0,0,10,,{\\pos(640,80)}车站\\N駅\n"
)


def _review_only_settings(tmp_path: Path, **values):
    # Only the review Agent is configured: nothing is drafted.
    return make_settings({
        "source_language": "ja",
        "target_language": "zh-CN",
        "agent": {"final": {"backend": "codex"}},
        "memory": tmp_path / "memory.md",
        "work_dir": tmp_path / "work",
        **values,
    })


def _existing_subtitle(tmp_path: Path, name: str, text: str = AEGISUB_ASS) -> Path:
    path = tmp_path / "subs" / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


def test_review_only_job_stages_copies_and_starts_reviewing(tmp_path: Path) -> None:
    original = _existing_subtitle(tmp_path, "movie.ja-zh-cn.ass")
    media = tmp_path / "movie.mkv"
    media.write_bytes(b"media")

    job = YakiFlowJob.for_review(
        [original], _review_only_settings(tmp_path), media=media
    )

    assert job.review_only
    assert job.backend is None
    assert job.db.job()["status"] == "reviewing"
    assert job.outputs == [job.work_dir / original.name]
    assert job.outputs[0].read_bytes() == original.read_bytes()
    assert original.read_bytes() == AEGISUB_ASS.encode("utf-8")
    assert job.media_path == media
    # A cue names a speaker, so the preview shows the speaker column.
    assert job.settings.diarization_enabled
    job.close()


def test_review_only_finalize_writes_edits_back_and_keeps_foreign_styling(
    tmp_path: Path,
) -> None:
    original = _existing_subtitle(tmp_path, "movie.ja-zh-cn.ass")
    job = YakiFlowJob.for_review([original], _review_only_settings(tmp_path))
    edited = AEGISUB_ASS.replace("{\\i1}你好{\\i0}", "{\\i1}您好{\\i0}")
    job.outputs[0].write_bytes(edited.encode("utf-8"))

    published = job.finalize_artifacts()

    assert published == [original]
    # Header, styles, comment, override tags, layer, and margins: all kept.
    assert original.read_bytes() == edited.encode("utf-8")
    job.close()


def test_review_only_repairs_disorder_in_a_file_yakiflow_wrote(
    tmp_path: Path,
) -> None:
    events = [
        AssEvent(0.0, 1.0, "", ("T:one", "one")),
        AssEvent(1.0, 2.0, "", ("T:two", "two")),
    ]
    original = _existing_subtitle(
        tmp_path, "movie.en-zh-cn.ass", render_events(events, (1080, 1920))
    )
    job = YakiFlowJob.for_review([original], _review_only_settings(tmp_path))
    job.outputs[0].write_text(
        render_events(events[::-1], (1080, 1920)), encoding="utf-8"
    )

    job.finalize_artifacts()

    # Repaired at the resolution the file was written at, not the default.
    assert original.read_text(encoding="utf-8") == render_events(
        events, (1080, 1920)
    )
    job.close()


def test_review_only_tolerates_defects_the_originals_already_had(
    tmp_path: Path,
) -> None:
    # The second, unnamed event overlaps the first.
    overlapping = AEGISUB_ASS.replace("0:00:02.50,Default", "0:00:03.50,Default")
    original = _existing_subtitle(tmp_path, "movie.ja-zh-cn.ass", overlapping)
    job = YakiFlowJob.for_review([original], _review_only_settings(tmp_path))
    edited = overlapping.replace("车站", "车站到了")
    job.outputs[0].write_text(edited, encoding="utf-8")

    job.finalize_artifacts()

    assert original.read_text(encoding="utf-8") == edited
    job.close()


def test_review_only_refuses_an_emptied_or_reencoded_review(tmp_path: Path) -> None:
    original = _existing_subtitle(tmp_path, "movie.ja-zh-cn.ass")
    job = YakiFlowJob.for_review([original], _review_only_settings(tmp_path))
    staged = job.outputs[0]

    staged.write_text(AEGISUB_ASS.split("Comment:")[0], encoding="utf-8")
    with pytest.raises(RuntimeError, match="contains no subtitle cues"):
        job.finalize_artifacts()
    reencoded = render_events([AssEvent(0.0, 1.0, "", ("字幕",))]).encode("gbk")
    staged.write_bytes(reencoded)
    with pytest.raises(RuntimeError, match="not valid UTF-8"):
        job.finalize_artifacts()

    assert staged.read_bytes() == reencoded
    assert original.read_bytes() == AEGISUB_ASS.encode("utf-8")
    assert job.db.job()["status"] == "reviewing"
    job.close()


def test_review_only_finalize_leaves_draft_snapshots_beside_the_original(
    tmp_path: Path,
) -> None:
    original = _existing_subtitle(tmp_path, "movie.ja-zh-cn.ass")
    # Neither was written by this review: the first is a concurrent run's
    # draft, the second is what a run would take for this job's own draft.
    drafts = [
        original.parent / "movie.draft.incomplete.ass",
        original.parent / f"{original.name}.draft.incomplete.ass",
    ]
    for draft in drafts:
        draft.write_text("someone else's draft\n", encoding="utf-8")
    job = YakiFlowJob.for_review([original], _review_only_settings(tmp_path))

    job.finalize_artifacts()

    assert all(draft.is_file() for draft in drafts)
    job.close()


def test_review_only_publishes_memory(tmp_path: Path) -> None:
    destination = tmp_path / "memory.md"
    destination.write_text("# Memory\n\n- original\n", encoding="utf-8")
    original = _existing_subtitle(tmp_path, "movie.ja-zh-cn.ass")
    job = YakiFlowJob.for_review([original], _review_only_settings(tmp_path))
    job.memory_path.write_text(
        "# Memory\n\n- original\n- 駅: render as 车站\n", encoding="utf-8"
    )

    job.finalize_artifacts()

    assert destination.read_text(encoding="utf-8").endswith("- 駅: render as 车站\n")
    job.close()


class NoProcessRunner:
    async def run(self, args, **kwargs):
        raise AssertionError(f"a review-only job ran {args}")


def test_review_only_job_resumes_into_review_without_the_pipeline(
    tmp_path: Path,
) -> None:
    original = _existing_subtitle(tmp_path, "movie.ja-zh-cn.ass")
    job = YakiFlowJob.for_review([original], _review_only_settings(tmp_path))
    staged = job.outputs[0]
    edited = AEGISUB_ASS.replace("车站", "车站到了")
    staged.write_text(edited, encoding="utf-8")
    work_dir = job.work_dir
    job.close()

    resumed = YakiFlowJob.from_workdir(work_dir, runner=NoProcessRunner())
    assert resumed.review_only and resumed.backend is None
    assert asyncio.run(resumed.run()) == [staged]
    assert resumed.finalize_artifacts() == [original]
    resumed.finish_review()

    assert original.read_text(encoding="utf-8") == edited
    resumed.close()


def test_half_staged_review_only_job_resumes_into_an_error(tmp_path: Path) -> None:
    original = _existing_subtitle(tmp_path, "movie.ja-zh-cn.ass")
    # Staging copied a file, then stopped before it recorded what it staged.
    job = YakiFlowJob(
        str(original),
        _review_only_settings(tmp_path),
        _review={"tolerated_problems": [], "canonical_play_res": {}},
    )
    work_dir = job.work_dir
    (work_dir / original.name).write_bytes(original.read_bytes())
    job.close()

    resumed = YakiFlowJob.from_workdir(work_dir, runner=NoProcessRunner())
    with pytest.raises(RuntimeError, match="missing staged subtitle files"):
        asyncio.run(resumed.run())
    resumed.close()


def test_review_only_rejects_unusable_inputs_before_creating_a_work_directory(
    tmp_path: Path,
) -> None:
    usable = _existing_subtitle(tmp_path, "movie.ja-zh-cn.ass")
    srt = tmp_path / "subs" / "movie.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nhello\n", encoding="utf-8")
    elsewhere = tmp_path / "other" / "movie.source.ass"
    elsewhere.parent.mkdir()
    elsewhere.write_text(AEGISUB_ASS, encoding="utf-8")
    empty = _existing_subtitle(
        tmp_path, "empty.ass", AEGISUB_ASS.split("Comment:")[0]
    )
    settings = _review_only_settings(tmp_path)

    for subtitles, media, error in (
        ([srt], None, "only ASS subtitles can be reviewed"),
        ([usable, elsewhere], None, "must be in one directory"),
        ([empty], None, "contains no subtitle cues"),
        ([usable], tmp_path / "absent.mkv", "media file does not exist"),
    ):
        with pytest.raises((ValueError, FileNotFoundError), match=error):
            YakiFlowJob.for_review(subtitles, settings, media=media)

    assert not (tmp_path / "work").exists()


def test_overlapping_translation_does_not_add_an_empty_progress_stage(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF-fake")
    settings = make_settings(
        source_language="auto",
        target_language="zh-CN",
        agent={"backend": "codex", "draft": {"batch_size": 1, "preceding_context": 1}},
        whisper={"model": model},
        memory=tmp_path / "memory.md",
        work_dir=tmp_path / "work",
        output_dir=tmp_path,
    )
    events = []
    backend = PipelineBackend()

    async def listener(event):
        events.append(event)

    job = YakiFlowJob(
        "input.mp4",
        settings,
        runner=PipelineRunner(),
        backend=backend,
        listener=listener,
    )
    artifact = MediaArtifact(MediaSource.parse("input.mp4"), audio)
    job._configure_progress()

    async def run_stages() -> None:
        cues = await job._transcription_stage(artifact)
        progress_after_transcription = events[-1].progress
        assert all(cue.translated for cue in cues)
        await job._translation_stage(artifact, cues)
        assert events[-1].progress == progress_after_transcription

    asyncio.run(run_stages())

    assert "translate" not in job._progress_plan.ranges
    assert not any(event.stage == "translate" for event in events)
    payloads = [
        json.loads(prompt.split("INPUT:\n", 1)[1]) for prompt in backend.prompts
    ]
    payload_by_cue = {payload["cues"][0]["id"]: payload for payload in payloads}
    assert [cue["id"] for cue in payload_by_cue["1"]["preceding_context"]] == []
    assert [cue["id"] for cue in payload_by_cue["2"]["preceding_context"]] == ["1"]
    job.close()


def test_authoritative_transcription_retranslates_provisional_stream_cues(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF-fake")
    backend = PipelineBackend()
    job = YakiFlowJob(
        "input.mp4",
        make_settings(
            source_language="auto",
            target_language="zh-CN",
            agent={"backend": "codex", "draft": {"batch_size": 20}},
            whisper={"model": model},
            memory=tmp_path / "memory.md",
            work_dir=tmp_path / "work",
            output_dir=tmp_path,
            stream=True,
        ),
        runner=PipelineRunner(),
        backend=backend,
    )
    job.db.upsert_cues([
        Cue("1", 0, 1, "unrelated live text", "T:unrelated live text"),
        Cue("2", 1, 2, "other live text", "T:other live text"),
    ])
    artifact = MediaArtifact(MediaSource.parse("input.mp4"), audio)
    job._configure_progress()

    async def run_stages() -> list[Cue]:
        transcribed = await job._transcription_stage(artifact)
        assert all(cue.translated is None for cue in transcribed)
        return await job._translation_stage(artifact, transcribed)

    result = asyncio.run(run_stages())

    assert [cue.source for cue in result] == ["hello", "world"]
    assert [cue.translated for cue in result] == ["T:hello", "T:world"]
    assert len(backend.prompts) == 1
    job.close()


def test_resumed_progress_scales_transcription_to_remaining_audio(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    audio = tmp_path / "reference.wav"
    with wave.open(str(audio), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(16_000)
        destination.writeframes(b"\0\0" * 160_000)
    settings = make_settings(
        source_language="auto",
        target_language="zh-CN",
        agent={"backend": "codex"},
        whisper={"model": model},
        memory=tmp_path / "memory.md",
        work_dir=tmp_path / "work",
        output_dir=tmp_path,
    )
    job = YakiFlowJob("input.mp4", settings, backend=PipelineBackend())
    job.db.add_artifact("reference_audio", audio)
    job.db.upsert_cues([Cue("1", 0, 9, "mostly complete", "已完成")])
    job.db.checkpoint("transcription_started", True)

    job._configure_progress()

    transcribe = job._progress_plan.ranges["transcribe"]
    align = job._progress_plan.ranges["align"]
    assert transcribe.span / align.span == pytest.approx((38 * 0.1) / 8)
    assert "acquire" not in job._progress_plan.ranges
    assert "translate" not in job._progress_plan.ranges
    job.close()


def test_memory_destination_change_must_be_accepted_before_finalize(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "memory.md"
    destination.write_text("# Memory\n\n- original\n", encoding="utf-8")
    settings = make_settings(
        source_language="en",
        target_language="zh-CN",
        agent={"backend": "codex"},
        memory=destination,
        work_dir=tmp_path / "work",
    )
    job = YakiFlowJob("input.mp4", settings, backend=PipelineBackend())
    job.memory_path.write_text(
        "# Memory\n\n- original\n- work-dir edit\n", encoding="utf-8"
    )
    destination.write_text(
        "# Memory\n\n- original\n- destination edit\n", encoding="utf-8"
    )

    with pytest.raises(MemoryDestinationConflict) as raised:
        job.finalize_artifacts()

    assert destination.read_text(encoding="utf-8").endswith("- destination edit\n")
    assert "- destination edit" in raised.value.diff()
    assert "+- destination edit" in raised.value.diff()

    job.memory_path.write_text(
        "# Memory\n\n- original\n- work-dir edit\n- destination edit\n",
        encoding="utf-8",
    )
    job.accept_memory_destination_change(raised.value)
    job.finalize_artifacts()

    assert "- work-dir edit" in destination.read_text(encoding="utf-8")
    assert "- destination edit" in destination.read_text(encoding="utf-8")
    job.close()


def test_interrupted_whisper_resumes_after_preserved_cues(tmp_path: Path) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF-audio")
    work_dir = tmp_path / "work"
    settings = make_settings(
        source_language="auto",
        target_language="zh-CN",
        agent={"backend": "codex"},
        whisper={"model": model},
        memory=tmp_path / "memory.md",
        work_dir=work_dir,
        output_dir=tmp_path,
    )
    artifact = MediaArtifact(MediaSource.parse("input.mp4"), audio)
    job = YakiFlowJob(
        "input.mp4",
        settings,
        runner=InterruptedWhisperRunner(),
        backend=PipelineBackend(),
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(job._transcription_stage(artifact))
    interrupted = job.db.list_cues(stable_only=True)
    assert [(cue.id, cue.source) for cue in interrupted] == [
        ("1", "hello"),
        ("2", "world"),
    ]
    job.db.upsert_cues([
        Cue(
            interrupted[0].id,
            interrupted[0].start,
            interrupted[0].end,
            interrupted[0].source,
            "你好",
        )
    ])
    preserved = job.db.list_cues(stable_only=True)
    assert job.db.get_checkpoint("transcription_started") is True
    job.db.set_status("interrupted", "interrupted")
    job.close()

    runner = ResumedWhisperRunner()
    resumed = YakiFlowJob.from_workdir(
        work_dir,
        runner=runner,
        backend=PipelineBackend(),
    )
    result = asyncio.run(resumed._transcription_stage(artifact))

    assert result[0] == preserved[0]
    assert result[1].translated == "T:world"
    assert [(cue.id, cue.start, cue.end, cue.source) for cue in result] == [
        ("1", 0.2, 1.5, "hello"),
        ("2", 1.5, 2.8, "world"),
        ("3", 3.0, 4.0, "again"),
    ]
    ffmpeg_call = next(call for call in runner.calls if call[0] == "ffmpeg")
    whisper_call = next(call for call in runner.calls if call[0] == "whisper-cli")
    assert ffmpeg_call[ffmpeg_call.index("-ss") + 1] == "2.8"
    assert whisper_call[whisper_call.index("-f") + 1] == str(
        work_dir / "whisper-resume.wav"
    )
    assert resumed.db.get_checkpoint("transcribed") is True
    resumed.close()


def test_missing_translations_are_sent_as_contiguous_agent_batches(
    tmp_path: Path,
) -> None:
    backend = PipelineBackend()
    settings = make_settings(
        source_language="auto",
        target_language="zh-CN",
        agent={
            "backend": "codex",
            "draft": {
                "batch_size": 2,
                "preceding_context": 2,
                "following_context": 2,
            },
        },
        whisper={"model": tmp_path / "model.bin"},
        memory=tmp_path / "memory.md",
        work_dir=tmp_path / "work",
        output_dir=tmp_path,
    )
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF-audio")
    artifact = MediaArtifact(MediaSource.parse("input.mp4"), audio)
    job = YakiFlowJob("input.mp4", settings, backend=backend)
    job.db.upsert_cues([
        Cue("1", 0, 1, "one", "T:one"),
        Cue("2", 1, 2, "two"),
        Cue("3", 2, 3, "three"),
        Cue("4", 3, 4, "four", "T:four"),
        Cue("5", 4, 5, "five"),
        Cue("6", 5, 6, "six"),
        Cue("7", 6, 7, "seven"),
    ])

    result = asyncio.run(job._translation_stage(artifact, job.db.list_cues()))

    payloads = [json.loads(prompt.split("INPUT:\n", 1)[1]) for prompt in backend.prompts]
    assert [[cue["id"] for cue in payload["cues"]] for payload in payloads] == [
        ["2", "3"],
        ["5", "6"],
        ["7"],
    ]
    assert [
        [cue["id"] for cue in payload["preceding_context"]] for payload in payloads
    ] == [
        ["1"],
        ["3", "4"],
        ["5", "6"],
    ]
    assert [
        [cue["id"] for cue in payload["following_context"]] for payload in payloads
    ] == [
        ["4", "5"],
        ["7"],
        [],
    ]
    assert all(cue.translated for cue in result)
    job.close()


def test_transcription_failure_cancels_draft_agent_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = BlockingBackend()

    class FailingTranscriber:
        name = "fake"

        def __init__(self, *args, **kwargs):
            pass

        async def transcribe(
            self,
            audio,
            on_event=None,
            on_progress=None,
            *,
            resume_from=(),
        ):
            assert on_event is not None
            await on_event(TranscriptEvent(Cue("cue", 0, 1, "hello"), final=False))
            await backend.started.wait()
            raise ValueError("transcription failed")

    monkeypatch.setattr(
        job_module, "make_transcriber", lambda *args, **kwargs: FailingTranscriber()
    )
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF-fake")
    settings = make_settings(
        source_language="auto",
        target_language="zh-CN",
        agent={
            "backend": "codex",
            "draft": {"model": "draft", "batch_size": 1},
            "final": {"model": "final"},
        },
        whisper={"model": tmp_path / "model.bin"},
        memory=tmp_path / "memory.md",
        work_dir=tmp_path / "work",
    )
    job = YakiFlowJob("input.mp4", settings, backend=backend)
    artifact = MediaArtifact(MediaSource.parse("input.mp4"), audio)

    async def run_stage() -> None:
        with pytest.raises(ValueError, match="transcription failed"):
            await job._transcription_stage(artifact)
        assert backend.cancelled.is_set()
        assert not job._agent_operations

    asyncio.run(run_stage())
    job.close()


def test_resuming_a_finished_job_does_not_republish_over_reviewed_files(
    tmp_path: Path,
) -> None:
    """Review edits live only in the published files, never in the database."""
    media = tmp_path / "movie.mp4"
    media.write_bytes(b"media")
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    work_dir = tmp_path / "work"
    output_dir = tmp_path / "out"
    settings = make_settings(
        source_language="auto",
        target_language="zh-CN",
        agent={"backend": "codex"},
        whisper={"model": model},
        memory=tmp_path / "memory.md",
        output_dir=output_dir,
        work_dir=work_dir,
    )
    job = YakiFlowJob(
        str(media), settings, runner=PipelineRunner(), backend=PipelineBackend()
    )
    reviewed = render_ass([Cue("1", 0, 1, "reviewed by Agent", "评审后的字幕")])
    staged = asyncio.run(job.run())[0]
    staged.write_text(reviewed, encoding="utf-8")
    published = job.finalize_artifacts()[0]
    job.finish_review()
    job.close()

    resumed = YakiFlowJob.from_workdir(work_dir, backend=PipelineBackend())
    assert resumed.is_finished
    with pytest.raises(RuntimeError, match="already complete"):
        asyncio.run(resumed.run())
    assert published.read_text(encoding="utf-8") == reviewed
    assert resumed.db.job()["status"] == "complete"
    resumed.close()


def test_alignment_none_extends_ends_but_leaves_crosstalk_alone(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF-fake")
    settings = make_settings(
        source_language="en",
        target_language="zh-CN",
        agent={"backend": "codex"},
        alignment={"backend": "none"},
        memory=tmp_path / "memory.md",
        work_dir=tmp_path / "work",
    )
    job = YakiFlowJob("input.mp4", settings, backend=PipelineBackend())
    # The first two speak over each other, so their ASR timing stands; the
    # third talks alone and still gets the shared end extension.
    cues = [
        Cue("1", 0.0, 4.0, "a", "甲", speaker="1"),
        Cue("2", 2.0, 6.0, "b", "乙", speaker="2"),
        Cue("3", 8.0, 9.0, "c", "丙", speaker="1"),
    ]
    job.db.upsert_cues(cues)
    artifact = MediaArtifact(MediaSource.parse("input.mp4"), audio)

    aligned = asyncio.run(job._alignment_stage(artifact, cues))

    assert [(cue.start, cue.end, cue.speaker) for cue in aligned] == [
        (0.0, 4.0, "1"),
        (2.0, 6.0, "2"),
        (8.0, 9.5, "1"),
    ]
    assert job.db.get_checkpoint("alignment_complete") is True
    assert job.alignment_result is not None
    assert job.alignment_result.backend == "none"
    job.close()


def test_whisperx_alignment_returns_diarized_crosstalk_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yakiflow.alignment as alignment_module

    class FakeWhisperX:
        def load_audio(self, _path):
            return [0.0] * (16_000 * 10)

        def nltk_load(self, _resource):
            return object()

        def load_align_model(self, **_kwargs):
            return object(), {"dictionary": "fake"}

        def align(self, segments, _model, _metadata, _audio, _device, **_kwargs):
            return {
                "segments": [{
                    "text": segments[0]["text"],
                    "words": [{"word": "c", "start": 0.3, "end": 1.1, "score": 0.9}],
                }]
            }

    monkeypatch.setattr(alignment_module, "import_module", lambda _name: FakeWhisperX())
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test")
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 160000)
    settings = make_settings(
        source_language="en",
        target_language="zh-CN",
        transcription={"backend": "elevenlabs"},
        agent={"backend": "codex"},
        alignment={"backend": "whisperx"},
        memory=tmp_path / "memory.md",
        work_dir=tmp_path / "work",
    )
    job = YakiFlowJob("input.mp4", settings, backend=PipelineBackend())
    assert job.settings.elevenlabs.diarize
    cues = [
        Cue("1", 1.0, 3.0, "a", "甲", speaker="1"),
        Cue("2", 2.0, 4.0, "b", "乙", speaker="2"),
        Cue("3", 8.0, 9.0, "c", "丙", speaker="1"),
    ]
    job.db.upsert_cues(cues)
    artifact = MediaArtifact(MediaSource.parse("input.mp4"), audio)

    aligned = asyncio.run(job._alignment_stage(artifact, cues))

    # Forced alignment moves the cue that stands alone and hands the crosstalk
    # pair back exactly as the ASR timed it, end extension included.
    assert [(cue.start, cue.end, cue.speaker) for cue in aligned] == [
        (1.0, 3.0, "1"),
        (2.0, 4.0, "2"),
        (pytest.approx(8.1), pytest.approx(9.4), "1"),
    ]
    assert job.alignment_result is not None
    assert job.alignment_result.backend == "whisperx"
    job.close()


def test_external_whisper_server_job_needs_no_local_model(tmp_path: Path) -> None:
    settings = make_settings(
        source_language="en",
        target_language="zh-CN",
        agent={"backend": "codex"},
        transcription={"backend": "whisper-server"},
        whisper={
            "server_url": "http://127.0.0.1:9999",
            "model": tmp_path / "definitely-missing.bin",
        },
        memory=tmp_path / "memory.md",
        work_dir=tmp_path / "work",
    )
    job = YakiFlowJob("input.mp4", settings, backend=PipelineBackend())
    # A missing custom model path is fatal for local Whisper, but an external
    # server owns its own model, so nothing is checked or downloaded.
    asyncio.run(job._ensure_model())
    job.close()

    local = make_settings(
        source_language="en",
        target_language="zh-CN",
        agent={"backend": "codex"},
        whisper={"model": tmp_path / "definitely-missing.bin"},
        memory=tmp_path / "memory.md",
        work_dir=tmp_path / "work2",
    )
    job = YakiFlowJob("input.mp4", local, backend=PipelineBackend())
    with pytest.raises(FileNotFoundError):
        asyncio.run(job._ensure_model())
    job.close()


def test_word_timeline_view_maintains_time_order_incrementally() -> None:
    from yakiflow.job import WordTimelineView

    preview = [
        Cue("preview-1", 0.0, 3.0, "one", metadata={"word_range": [0, 5]}),
        Cue("preview-2", 4.0, 7.0, "two", metadata={"word_range": [6, 11]}),
        Cue("preview-3", 8.0, 11.0, "three", metadata={"word_range": [12, 17]}),
    ]
    view = WordTimelineView([], preview)
    assert [cue.id for cue in view.cues()] == ["preview-1", "preview-2", "preview-3"]

    # A middle batch lands first: it retires exactly the preview rows whose
    # words it covers and enters at its time position, no re-sort involved.
    view.apply([
        Cue("w9-11", 5.5, 7.0, "two b", "二乙", metadata={"word_range": [9, 11]}),
        Cue("w6-8", 4.0, 5.5, "two a", "二甲", metadata={"word_range": [6, 8]}),
    ])
    assert [cue.id for cue in view.cues()] == [
        "preview-1", "w6-8", "w9-11", "preview-3",
    ]

    # A junction repair replaces both halves with one combined cue.
    view.apply(
        [Cue("w6-11", 4.0, 7.0, "two", "二", metadata={"word_range": [6, 11]})],
        ["w6-8", "w9-11"],
    )
    assert [cue.id for cue in view.cues()] == ["preview-1", "w6-11", "preview-3"]

    view.apply([Cue("w0-5", 0.0, 3.0, "one", "一", metadata={"word_range": [0, 5]})])
    view.apply([Cue("w12-17", 8.0, 11.0, "three", "三", metadata={"word_range": [12, 17]})])
    assert [cue.id for cue in view.cues()] == ["w0-5", "w6-11", "w12-17"]

    # Seeding treats already-covered preview rows as retired.
    seeded = WordTimelineView(view.cues(), preview)
    assert [cue.id for cue in seeded.cues()] == ["w0-5", "w6-11", "w12-17"]


def test_word_mode_job_builds_the_timeline_from_agent_segmentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from yakiflow.elevenlabs import ElevenLabsTranscriber

    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 16000)

    class FakeSpeechToText:
        async def convert(self, **kwargs):
            class Response:
                language_code = "en"
                words = [
                    {"type": "word", "text": "How ", "start": 0.0, "end": 0.4, "speaker_id": "speaker_1"},
                    {"type": "word", "text": "are ", "start": 0.5, "end": 0.9, "speaker_id": "speaker_1"},
                    {"type": "word", "text": "you", "start": 1.0, "end": 1.4, "speaker_id": "speaker_1"},
                    {"type": "word", "text": "Fine", "start": 1.2, "end": 1.8, "speaker_id": "speaker_2"},
                ]
            return Response()

    class FakeClient:
        speech_to_text = FakeSpeechToText()

    class WordBackend(AgentBackend):
        name = "fake"

        async def invoke_with_trace(
            self, prompt, *, system="", model, effort, schema, on_event=None
        ):
            payload = json.loads(prompt.split("INPUT:\n", 1)[1])
            words = payload["words"]
            cues: list[dict] = []
            current: list[dict] = []
            for word in words:
                if current and current[-1].get("s") != word.get("s"):
                    cues.append({
                        "first_word": current[0]["i"],
                        "last_word": current[-1]["i"],
                        "source": "".join(w["w"] for w in current).strip(),
                        "translated": "T:" + "".join(w["w"] for w in current).strip(),
                    })
                    current = []
                current.append(word)
            if current:
                cues.append({
                    "first_word": current[0]["i"],
                    "last_word": current[-1]["i"],
                    "source": "".join(w["w"] for w in current).strip(),
                    "translated": "T:" + "".join(w["w"] for w in current).strip(),
                })
            return {"cues": cues}

    monkeypatch.setattr(
        job_module,
        "make_transcriber",
        lambda settings, work_dir, db, runner=None: ElevenLabsTranscriber(
            settings, work_dir, db, runner, client_factory=lambda: FakeClient()
        ),
    )
    settings = make_settings(
        source_language="en",
        target_language="zh-CN",
        transcription={"backend": "elevenlabs"},
        agent={"backend": "codex", "draft": {"model": "draft"}},
        memory=tmp_path / "memory.md",
        work_dir=tmp_path / "work",
    )
    job = YakiFlowJob("input.mp4", settings, backend=WordBackend())
    # The alignment default for word-level backends is none.
    assert job.settings.alignment.backend == "none"
    events: list[JobEvent] = []

    async def listener(event: JobEvent) -> None:
        events.append(event)

    job.listener = listener
    artifact = MediaArtifact(MediaSource.parse("input.mp4"), audio)

    async def run_stages() -> list[Cue]:
        cues = await job._transcription_stage(artifact)
        cues = await job._translation_stage(artifact, cues)
        return await job._alignment_stage(artifact, cues)

    final = asyncio.run(run_stages())

    # Word-mode cue IDs churn as the agent supersedes the preview, so partial
    # draft results reach the TUI as whole-view timeline replacements.
    views = [event.cues for event in events if event.cues is not None]
    assert views, "no timeline-replaced event carried a replacement view"
    assert any(
        all(cue.translated for cue in view) for view in views if view
    ), "no view showed the drafted batch before the stage finished"
    assert [cue.id for cue in views[-1]] == ["1", "2"]

    assert job.db.get_checkpoint("detected_source_language") == "en"
    assert len(job.db.list_transcript_words()) == 4
    assert [(cue.speaker, cue.source, cue.translated) for cue in final] == [
        ("1", "How are you", "T:How are you"),
        ("2", "Fine", "T:Fine"),
    ]
    # The crosstalk overlap survives all the way through alignment none, and
    # the pair keeps the word times verbatim: the WAV above runs 1.0 s while
    # the words run to 1.8 s, so a clamp to the audio duration would flatten
    # both cues onto one end and erase the overlap.
    assert final[1].start < final[0].end
    assert [(cue.start, cue.end) for cue in final] == [(0.0, 1.4), (1.2, 1.8)]
    assert job.db.get_checkpoint("alignment_complete") is True
    job.close()
