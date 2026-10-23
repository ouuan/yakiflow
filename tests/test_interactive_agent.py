import asyncio
from pathlib import Path

import pytest

import yakiflow.interactive_agent as interactive_agent
from conftest import make_settings
from yakiflow.config import Settings
from yakiflow.interactive_agent import (
    build_interactive_command,
    build_interactive_prompt,
    build_memory_conflict_prompt,
    review_open_argv,
    start_agent_file_display,
)
from yakiflow.process import ProcessResult


@pytest.mark.parametrize(
    ("output_mode", "expected", "required_detail"),
    [
        (
            "source",
            "final deliverable is a source-language monolingual subtitle",
            "Do not add a translation or a second language line",
        ),
        (
            "translated",
            "final deliverable is a target-language monolingual subtitle",
            "do not add source-language lines",
        ),
        (
            "bilingual",
            "final deliverable is a bilingual subtitle",
            "Fix both clear source transcription problems and target translation problems",
        ),
        (
            "all",
            "final deliverables include source-language monolingual, target-language monolingual, and bilingual subtitles",
            "Mirror every applicable text and timing correction across all three artifacts",
        ),
    ],
)
def test_interactive_prompt_scopes_review_to_final_output(
    tmp_path: Path,
    output_mode: str,
    expected: str,
    required_detail: str,
) -> None:
    prompt = build_interactive_prompt(
        Settings(
            source_language="en",
            target_language="zh-CN",
            output_mode=output_mode,
        ),
        tmp_path,
        [tmp_path / "movie.ass"],
    )

    normalized = " ".join(prompt.split())
    assert expected in normalized
    assert required_detail in normalized
    assert "source/translation evidence applicable to the output-specific scope" in normalized
    assert "Whenever you point out or discuss a specific subtitle" in normalized
    assert "include its start and end timestamp" in normalized
    assert "0:01:23.45 - 0:01:25.00" in normalized


def test_interactive_prompt_repeats_every_review_cue_and_does_not_trust_asr_alone(
    tmp_path: Path,
) -> None:
    prompt = build_interactive_prompt(
        Settings(source_language="en", target_language="zh-CN"),
        tmp_path,
    )

    normalized = " ".join(prompt.split())
    assert "raw ASR result is evidence, never proof" in normalized
    assert "do not declare a cue correct solely because ASR produced that wording" in normalized
    assert "semantically abrupt or out of place" in normalized
    assert "actively consider similarly pronounced words" in normalized
    assert "ask the user to decide" in normalized
    assert "Whenever you ask the user for review feedback, confirmation, or a decision" in normalized
    assert "list every related subtitle cue in that message, one entry per cue" in normalized
    assert "Repeat the complete list every time" in normalized
    assert "even when the same cues were listed earlier in the conversation" in normalized
    assert "The final change summary presented for the user's review must also repeat every related subtitle cue" in normalized


@pytest.mark.parametrize(
    ("output_mode", "expects_phrasing_scope"),
    [
        ("source", False),
        ("translated", True),
        ("bilingual", True),
        ("all", True),
    ],
)
def test_interactive_prompt_requires_natural_target_phrasing(
    tmp_path: Path,
    output_mode: str,
    expects_phrasing_scope: bool,
) -> None:
    prompt = build_interactive_prompt(
        Settings(
            source_language="en",
            target_language="zh-CN",
            output_mode=output_mode,
        ),
        tmp_path,
    )

    normalized = " ".join(prompt.split())
    assert ("Phrasing quality for the target language" in normalized) is expects_phrasing_scope
    assert ("as if it had been written in that language" in normalized) is expects_phrasing_scope
    assert ("other translationese" in normalized) is expects_phrasing_scope
    assert ("wording a native speaker would actually use" in normalized) is expects_phrasing_scope
    assert ("Do not change the meaning, speaker intent, or tone" in normalized) is expects_phrasing_scope


@pytest.mark.parametrize(
    ("output_mode", "expects_merge_scope", "expects_mirroring"),
    [
        ("source", False, False),
        ("translated", True, False),
        ("bilingual", True, True),
        ("all", True, True),
    ],
)
def test_interactive_prompt_allows_merging_cues_a_translation_cannot_split(
    tmp_path: Path,
    output_mode: str,
    expects_merge_scope: bool,
    expects_mirroring: bool,
) -> None:
    prompt = build_interactive_prompt(
        Settings(
            source_language="en",
            target_language="zh-CN",
            output_mode=output_mode,
        ),
        tmp_path,
    )

    normalized = " ".join(prompt.split())
    assert ("Merging cues the target language cannot keep apart" in normalized) is expects_merge_scope
    assert ("no single translated cue can carry a complete" in normalized) is expects_merge_scope
    assert ("they belong to the same sentence and the same speaker" in normalized) is expects_merge_scope
    assert ("read comfortably within its on-screen time" in normalized) is expects_merge_scope
    assert ("they are not separated by a long pause" in normalized) is expects_merge_scope
    assert (
        "extend one Dialogue event's times to cover the whole span and delete the other Dialogue line"
        in normalized
    ) is expects_merge_scope
    assert ("Merging is the exception, not the default" in normalized) is expects_merge_scope
    assert ("Apply every merge to the source side" in normalized) is expects_mirroring
    # Renumbering is meaningless in ASS and must not be prescribed anymore.
    assert "Renumber" not in prompt
    assert "renumber" not in prompt


@pytest.mark.parametrize("output_mode", ["source", "translated", "bilingual", "all"])
def test_interactive_prompt_states_the_ass_format_contract(
    tmp_path: Path,
    output_mode: str,
) -> None:
    prompt = build_interactive_prompt(
        Settings(
            source_language="en",
            target_language="zh-CN",
            output_mode=output_mode,
        ),
        tmp_path,
    )

    normalized = " ".join(prompt.split())
    assert "ASS (Advanced SubStation Alpha) documents" in normalized
    assert "editing surface is the `Dialogue:` lines in the `[Events]` section" in normalized
    assert "regenerated wholesale at publish time" in normalized
    assert "Name field is the diarized speaker label" in normalized
    assert "leave it unchanged unless the user asks" in normalized
    assert "written as `\\N` in the Text field" in normalized
    assert (
        "translation first and source second, separated by `\\N`" in normalized
    ) is (output_mode in {"bilingual", "all"})
    assert (
        "All three artifacts must keep the same event count and identical event timings and Name fields"
        in normalized
    ) is (output_mode == "all")


@pytest.mark.parametrize("output_mode", ["source", "translated", "bilingual", "all"])
def test_interactive_prompt_leaves_mechanical_rules_to_publish_validation(
    tmp_path: Path,
    output_mode: str,
) -> None:
    prompt = build_interactive_prompt(
        Settings(
            source_language="en",
            target_language="zh-CN",
            output_mode=output_mode,
        ),
        tmp_path,
        [tmp_path / "movie.ass"],
    )

    # Publish-time validation code is the sole authority for the mechanical
    # invariants, so the prompt must not restate them.
    assert "cue numbers running 1..N" not in prompt
    assert "Renumber the whole file" not in prompt
    assert "HH:MM:SS,mmm" not in prompt
    assert "never end before they start" not in prompt
    assert "no cue left without text" not in prompt
    # The deliverable is no longer described as SRT anywhere.
    assert "SRT" not in prompt


@pytest.mark.parametrize("output_mode", ["source", "translated", "bilingual", "all"])
def test_interactive_prompt_permits_cross_speaker_overlaps(
    tmp_path: Path,
    output_mode: str,
) -> None:
    prompt = build_interactive_prompt(
        Settings(
            source_language="en",
            target_language="zh-CN",
            output_mode=output_mode,
        ),
        tmp_path,
    )

    normalized = " ".join(prompt.split())
    assert "Cues of two different named speakers may legitimately overlap in time" in normalized
    assert "simultaneous speech, not an error" in normalized
    assert "never merge, delete, or retime cues just to remove such an overlap" in normalized


def test_interactive_prompt_lists_previous_problems_verbatim(tmp_path: Path) -> None:
    problems = (
        "cue 12 ends before it starts (0:01:10.00 - 0:01:08.50)",
        "empty Dialogue text at 0:03:44.10",
    )

    prompt = build_interactive_prompt(
        Settings(source_language="en", target_language="zh-CN"),
        tmp_path,
        previous_problems=problems,
    )

    normalized = " ".join(prompt.split())
    assert "previous review session left these publish-blocking defects" in normalized
    assert "YakiFlow will re-check them when it publishes" in normalized
    for problem in problems:
        assert f"- {problem}" in prompt


def test_interactive_prompt_omits_previous_problems_section_when_empty(
    tmp_path: Path,
) -> None:
    prompt = build_interactive_prompt(
        Settings(source_language="en", target_language="zh-CN"),
        tmp_path,
    )

    assert "publish-blocking defects" not in prompt
    assert "previous review session" not in prompt


def test_interactive_prompt_keeps_everything_but_subtitles_and_memory_read_only(
    tmp_path: Path,
) -> None:
    prompt = build_interactive_prompt(
        Settings(source_language="en", target_language="zh-CN"),
        tmp_path,
    )

    normalized = " ".join(prompt.split())
    assert "Exactly two kinds of file are yours to edit" in normalized
    assert str(tmp_path / "memory.md") in normalized
    assert "read-only evidence" in normalized
    assert "can corrupt the job's resume state" in normalized
    assert "treat a missing one as empty memory" in normalized


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        (Settings(source_language="ja", target_language="zh-CN"), "zh-CN"),
        (Settings(source_language="ja", output_mode="source"), "ja"),
    ],
)
def test_interactive_prompt_defaults_the_conversation_to_the_target_language(
    tmp_path: Path,
    settings: Settings,
    expected: str,
) -> None:
    prompt = build_interactive_prompt(settings, tmp_path)

    normalized = " ".join(prompt.split())
    assert f"Write to the user in {expected}, including the opening review report" in normalized
    assert "switch to that language and keep using it" in normalized


def test_interactive_prompt_falls_back_to_the_user_language_without_configured_languages(
    tmp_path: Path,
) -> None:
    prompt = build_interactive_prompt(Settings(output_mode="source"), tmp_path)

    normalized = " ".join(prompt.split())
    assert "Write to the user in whatever language they use" in normalized


def test_interactive_prompt_requires_proactive_novel_memory_candidates(
    tmp_path: Path,
) -> None:
    prompt = build_interactive_prompt(
        Settings(source_language="en", target_language="zh-CN"),
        tmp_path,
    )

    normalized = " ".join(prompt.split())
    assert "from your own review" in normalized
    assert "do not wait for a comment" in normalized
    assert "canonical names, proper nouns, domain terms, acronyms" in normalized
    assert "recurring ASR corrections or domain meanings" in normalized
    assert "durable translation and style preferences" in normalized
    assert "stable project or series context" in normalized
    assert "Exclude duplicates of existing memory" in normalized
    assert "obtain explicit approval" in normalized


def test_interactive_prompt_rechecks_subtitles_after_memory_changes(
    tmp_path: Path,
) -> None:
    prompt = build_interactive_prompt(
        Settings(source_language="en", target_language="zh-CN"),
        tmp_path,
    )

    normalized = " ".join(prompt.split())
    assert "Whenever the user asks to add, remove, or change anything" in normalized
    assert "re-check every staged subtitle artifact against the resulting memory" in normalized
    assert "do not stop after editing memory" in normalized


def test_interactive_prompt_lists_read_only_context_files(tmp_path: Path) -> None:
    context_files = (
        tmp_path / "context" / "danmaku.xml",
        tmp_path / "context" / "speaker-notes.txt",
    )

    prompt = build_interactive_prompt(
        Settings(source_language="ja", target_language="zh-CN"),
        tmp_path,
        context_files=context_files,
    )

    assert str(context_files[0]) in prompt
    assert str(context_files[1]) in prompt
    assert "supporting context" in prompt
    assert "do not edit or publish them" in prompt


def test_memory_conflict_prompt_scopes_agent_to_staged_memory(tmp_path: Path) -> None:
    destination = tmp_path / "durable" / "memory.md"
    prompt = build_memory_conflict_prompt(
        tmp_path / "work",
        destination,
        "--- previous\n+++ current\n-old\n+new\n",
    )

    assert str(tmp_path / "work" / "memory.md") in prompt
    assert str(destination) in prompt
    assert "-old\n+new" in prompt
    assert "do not edit the destination" in prompt
    assert "avoid duplicate memory entries" in " ".join(prompt.split())


def test_review_open_command_expands_job_path_placeholders(tmp_path: Path) -> None:
    work_dir = tmp_path / "work dir"
    subtitle = work_dir / "movie file.srt"

    assert review_open_argv(
        "code --reuse-window {workdir} {srt} {memory}", subtitle, work_dir
    ) == [
        "code",
        "--reuse-window",
        str(work_dir),
        str(subtitle),
        str(work_dir / "memory.md"),
    ]


@pytest.mark.parametrize(
    "subtitle_placeholder", ["{subtitle}", "{srt}", "{file}", "{srt_file}"]
)
def test_review_open_command_supports_subtitle_aliases_without_appending(
    tmp_path: Path, subtitle_placeholder: str
) -> None:
    subtitle = tmp_path / "movie.ass"

    assert review_open_argv(f"editor {subtitle_placeholder}", subtitle, tmp_path) == [
        "editor",
        str(subtitle),
    ]


def test_review_open_command_supports_memory_file_alias(tmp_path: Path) -> None:
    subtitle = tmp_path / "movie.ass"

    assert review_open_argv("editor {memory_file}", subtitle, tmp_path) == [
        "editor",
        str(tmp_path / "memory.md"),
        str(subtitle),
    ]


def test_review_open_command_appends_subtitle_when_no_subtitle_placeholder(
    tmp_path: Path,
) -> None:
    subtitle = tmp_path / "movie.ass"

    assert review_open_argv("editor {workdir}", subtitle, tmp_path) == [
        "editor",
        str(tmp_path),
        str(subtitle),
    ]


def test_both_display_mode_opens_external_command_and_tmux_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subtitle = tmp_path / "movie.srt"
    subtitle.write_text("", encoding="utf-8")
    monkeypatch.setenv("TMUX", "test")
    monkeypatch.setattr(interactive_agent, "_source_media", lambda _work_dir: None)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class Process:
        returncode = 0

        async def wait(self) -> None:
            return None

    async def fake_create_subprocess_exec(*args, **kwargs):
        calls.append((args, kwargs))
        return Process()

    monkeypatch.setattr(
        interactive_agent.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    display = asyncio.run(
        start_agent_file_display(
            make_settings(
                review={"display_mode": "both", "open_command": "editor {srt}"},
            ),
            tmp_path,
            [subtitle],
        )
    )

    assert calls[0][0] == ("editor", str(subtitle))
    assert calls[1][0][:2] == ("tmux", "split-window")
    assert "yakiflow.subtitle_preview" in calls[1][0]
    assert display.marker == tmp_path / ".agent-display-done"


def test_preview_command_carries_the_speaker_column_flag(tmp_path: Path) -> None:
    subtitle = tmp_path / "movie.ass"

    diarized = make_settings(
        transcription={"backend": "elevenlabs"}, elevenlabs={"api_key": "k"}
    )
    plain = make_settings(
        transcription={"backend": "elevenlabs"},
        elevenlabs={"api_key": "k", "diarize": False},
    )
    marker = tmp_path / ".agent-display-done"

    assert interactive_agent._preview_command(
        subtitle, marker, None, None, diarized.diarization_enabled
    )[-1] == "1"
    assert interactive_agent._preview_command(
        subtitle, marker, None, None, plain.diarization_enabled
    )[-1] == ""


def test_interactive_commands_append_final_extra_options() -> None:
    codex = build_interactive_command(
        make_settings(
            agent={
                "final": {
                    "backend": "codex",
                    "model": "final-codex",
                    "effort": "high",
                    "extra_options": ["-c", "service_tier=fast"],
                },
            },
        ),
        "codex prompt",
    )
    claude = build_interactive_command(
        make_settings(
            agent={
                "final": {
                    "backend": "claude",
                    "model": "final-claude",
                    "effort": "medium",
                    "extra_options": ["--permission-mode", "plan"],
                },
            },
        ),
        "claude prompt",
    )

    assert codex == [
        "codex",
        "--sandbox",
        "workspace-write",
        "--model",
        "final-codex",
        "--config",
        'model_reasoning_effort="high"',
        "-c",
        "service_tier=fast",
        "codex prompt",
    ]
    assert claude == [
        "claude",
        "--model",
        "final-claude",
        "--effort",
        "medium",
        "--permission-mode",
        "plan",
        "claude prompt",
    ]


def test_interactive_agent_auto_opens_video_before_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TMUX", "test")
    media = tmp_path / "movie.mp4"
    subtitle = tmp_path / "movie.srt"
    media.write_bytes(b"media")
    subtitle.write_text("", encoding="utf-8")
    calls: list[tuple[str | None, Path | None, Path | None, Path | None]] = []

    monkeypatch.setattr(
        interactive_agent,
        "_source_media",
        lambda _work_dir: media,
    )

    def fake_open_media(command, media_path, subtitle_path, *, cwd=None):
        calls.append((command, media_path, subtitle_path, cwd))
        return True

    monkeypatch.setattr(interactive_agent, "open_media", fake_open_media)

    class Runner:
        async def run_interactive(self, command, *, cwd=None):
            assert calls == [
                (Settings().review.video_open_command, media, subtitle, tmp_path)
            ]
            return ProcessResult(tuple(command), 0, "", "")

    result = asyncio.run(
        interactive_agent.run_interactive_agent(
            make_settings(
                review={"auto_open_video": True},
                agent={"final": {"backend": "codex"}},
            ),
            tmp_path,
            [subtitle],
            runner=Runner(),
        )
    )

    assert result.returncode == 0


def test_interactive_agent_does_not_auto_open_video_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TMUX", "test")
    monkeypatch.setattr(interactive_agent, "_source_media", lambda _work_dir: None)
    opened = False

    def fake_open_media(*_args, **_kwargs):
        nonlocal opened
        opened = True
        return True

    monkeypatch.setattr(interactive_agent, "open_media", fake_open_media)

    class Runner:
        async def run_interactive(self, command, *, cwd=None):
            return ProcessResult(tuple(command), 0, "", "")

    asyncio.run(
        interactive_agent.run_interactive_agent(
            make_settings(agent={"final": {"backend": "codex"}}),
            tmp_path,
            [tmp_path / "movie.srt"],
            runner=Runner(),
        )
    )

    assert not opened


def test_interactive_agent_forwards_previous_problems_into_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TMUX", "test")
    monkeypatch.setattr(interactive_agent, "_source_media", lambda _work_dir: None)
    problems = ("cue 7 has no text", "cue 9 ends before it starts")
    prompts: list[str] = []

    class Runner:
        async def run_interactive(self, command, *, cwd=None):
            prompts.append(command[-1])
            return ProcessResult(tuple(command), 0, "", "")

    asyncio.run(
        interactive_agent.run_interactive_agent(
            make_settings(agent={"final": {"backend": "codex"}}),
            tmp_path,
            [tmp_path / "movie.ass"],
            previous_problems=problems,
            runner=Runner(),
        )
    )

    assert len(prompts) == 1
    assert "publish-blocking defects" in prompts[0]
    for problem in problems:
        assert f"- {problem}" in prompts[0]
