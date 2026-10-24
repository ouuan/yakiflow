import asyncio
import json
from pathlib import Path

import pytest

from conftest import make_settings
from yakiflow.database import JobDatabase
from yakiflow.models import AgentTraceEvent, Cue
from yakiflow.process import ProcessError, ProcessResult
from yakiflow.translation import (
    AgentBackend,
    ClaudeBackend,
    CodexBackend,
    GeminiBackend,
    TranslationPipeline,
    claude_trace_events,
    codex_trace_events,
    gemini_trace_events,
)


MEMORY = """## 人名

- ちゅんるん → 麻圆
  - ASR 纠错：純潤/旬論 → ちゅんるん
"""


class RecordingAwaitable:
    def __init__(self, values: list, value: object) -> None:
        self.values = values
        self.value = value

    def __await__(self):
        self.values.append(self.value)
        if False:
            yield
        return None


class FakeBackend(AgentBackend):
    name = "fake"

    def __init__(self):
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.schemas: list[dict] = []

    async def invoke_with_trace(
        self, prompt, *, system="", model, effort, schema, on_event=None
    ):
        self.prompts.append(prompt)
        self.systems.append(system)
        self.schemas.append(schema)
        response = {
            "cues": [{"id": "c1", "source": "agent source", "translated": "代理译文"}],
        }
        return response


class EchoBackend(AgentBackend):
    """Translate whatever cues the prompt actually asked for."""

    name = "echo"

    def __init__(self):
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
                {"id": cue["id"], "source": cue["source"], "translated": f"T:{cue['id']}"}
                for cue in payload["cues"]
            ],
        }


class SequenceBackend(AgentBackend):
    name = "sequence"

    def __init__(self, translations: list[str]):
        self.translations = translations
        self.calls = 0

    async def invoke_with_trace(
        self, prompt, *, system="", model, effort, schema, on_event=None
    ):
        translated = self.translations[min(self.calls, len(self.translations) - 1)]
        self.calls += 1
        return {
            "cues": [{"id": "c1", "source": "hello", "translated": translated}],
        }


class TimeoutThenSuccessBackend(AgentBackend):
    name = "timeout"

    def __init__(self):
        self.calls = 0

    async def invoke_with_trace(
        self, prompt, *, system="", model, effort, schema, on_event=None
    ):
        self.calls += 1
        if self.calls == 1:
            await asyncio.sleep(1)
        return {
            "cues": [{"id": "c1", "source": "hello", "translated": "你好"}],
        }


def test_cli_trace_adapters_normalize_messages_and_tools() -> None:
    started = codex_trace_events({
        "type": "item.started",
        "item": {
            "id": "tool-1",
            "type": "command_execution",
            "command": "rg hello subtitles.srt",
        },
    })
    completed = codex_trace_events({
        "type": "item.completed",
        "item": {
            "id": "tool-1",
            "type": "command_execution",
            "command": "rg hello subtitles.srt",
            "exit_code": 0,
            "aggregated_output": "1:hello",
        },
    })
    hidden_json = codex_trace_events({
        "type": "item.completed",
        "item": {
            "id": "message-1",
            "type": "agent_message",
            "text": '{"cues": []}',
        },
    })
    claude = claude_trace_events({
        "type": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": "Checking terminology."},
                {"type": "tool_use", "id": "tool-2", "name": "Read", "input": {"file": "memory.md"}},
            ]
        },
    })

    assert started[0].kind == "tool_start"
    assert completed[0].kind == "tool_result"
    assert completed[0].detail == "1:hello"
    assert hidden_json == []
    assert [event.kind for event in claude] == ["agent_message", "tool_start"]
    assert claude[1].detail == "file: memory.md"


def test_cli_backends_stream_trace_while_returning_structured_result(tmp_path: Path) -> None:
    schema = {
        "type": "object",
        "properties": {"cues": {"type": "array"}},
        "required": ["cues"],
    }
    expected = {"cues": [{"id": "c1", "source": "hello", "translated": "你好"}]}

    class CodexRunner:
        async def run(self, args, *, on_line=None, **kwargs):
            assert "--json" in args
            output = Path(args[args.index("--output-last-message") + 1])
            output.write_text(json.dumps(expected))
            assert on_line is not None
            await on_line("stdout", json.dumps({
                "type": "item.started",
                "item": {"id": "tool", "type": "command_execution", "command": "pwd"},
            }))
            return ProcessResult(tuple(str(arg) for arg in args), 0, "", "")

    class ClaudeRunner:
        async def run(self, args, *, on_line=None, **kwargs):
            assert args[args.index("--output-format") + 1] == "stream-json"
            assert "--json-schema" in args
            assert on_line is not None
            await on_line("stdout", json.dumps({
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Working on it."}]},
            }))
            await on_line("stdout", json.dumps({
                "type": "result", "structured_output": expected,
            }))
            return ProcessResult(tuple(str(arg) for arg in args), 0, "", "")

    async def exercise():
        codex_events = []
        claude_events = []
        codex = await CodexBackend(tmp_path, CodexRunner()).invoke_with_trace(
            "prompt", model="model", effort="medium", schema=schema,
            on_event=lambda event: RecordingAwaitable(codex_events, event),
        )
        claude = await ClaudeBackend(tmp_path, ClaudeRunner()).invoke_with_trace(
            "prompt", model="model", effort="medium", schema=schema,
            on_event=lambda event: RecordingAwaitable(claude_events, event),
        )
        return codex, claude, codex_events, claude_events

    codex, claude, codex_events, claude_events = asyncio.run(exercise())
    assert codex == expected
    assert claude == expected
    assert codex_events[0].kind == "tool_start"
    assert claude_events[0].kind == "agent_message"


def test_draft_backend_commands_append_only_matching_options(tmp_path: Path) -> None:
    schema = {"type": "object"}
    expected = {"cues": []}
    commands: dict[str, list[object]] = {}
    stdin: dict[str, bytes] = {}
    staged: dict[str, str] = {}

    class CodexRunner:
        async def run(self, args, **kwargs):
            commands["codex"] = list(args)
            stdin["codex"] = kwargs.get("stdin", b"")
            output = Path(args[args.index("--output-last-message") + 1])
            output.write_text(json.dumps(expected))
            return ProcessResult(tuple(str(arg) for arg in args), 0, "", "")

    class ClaudeRunner:
        async def run(self, args, *, on_line=None, **kwargs):
            commands["claude"] = list(args)
            stdin["claude"] = kwargs.get("stdin", b"")
            index = list(args).index("--system-prompt-file")
            staged["system"] = Path(args[index + 1]).read_text(encoding="utf-8")
            assert on_line is not None
            await on_line(
                "stdout",
                json.dumps({"type": "result", "structured_output": expected}),
            )
            return ProcessResult(tuple(str(arg) for arg in args), 0, "", "")

    async def exercise() -> None:
        await CodexBackend(
            tmp_path, CodexRunner(), ("-c", "service_tier=fast")
        ).invoke_with_trace(
            "prompt", system="stable guidance", model="draft-codex",
            effort="low", schema=schema,
        )
        await ClaudeBackend(
            tmp_path, ClaudeRunner(), ("--permission-mode", "plan")
        ).invoke_with_trace(
            "prompt", system="stable guidance", model="draft-claude",
            effort="medium", schema=schema,
        )

    asyncio.run(exercise())

    codex = commands["codex"]
    assert codex[-3:] == ["-c", "service_tier=fast", "-"]
    assert "--permission-mode" not in codex
    # Codex has no system-prompt flag, so the stable text leads the stdin prompt
    # and its automatic prefix matching does the rest.
    assert "--system-prompt-file" not in codex
    assert stdin["codex"] == b"stable guidance\nprompt"

    claude = commands["claude"]
    assert staged["system"] == "stable guidance"
    assert stdin["claude"] == b"prompt"
    # `--tools` is variadic, so an empty value has to be followed by a flag
    # rather than by a caller-supplied token it would otherwise swallow.
    assert claude[claude.index("--tools") + 1:] == [
        "",
        "--no-session-persistence",
        "--permission-mode",
        "plan",
    ]
    assert "service_tier=fast" not in claude
    # The staged system prompt is written for the call and cleaned up after it.
    assert not list(tmp_path.glob("agent-system-*.md"))


def test_gemini_trace_adapter_normalizes_tools_and_errors() -> None:
    started = gemini_trace_events({
        "type": "tool_use",
        "tool_name": "read_file",
        "tool_id": "tool-1",
        "parameters": {"file_path": "memory.md"},
    })
    completed = gemini_trace_events({
        "type": "tool_result", "tool_id": "tool-1", "status": "success",
        "output": "## Names",
    })
    failed = gemini_trace_events({
        "type": "tool_result", "tool_id": "tool-2", "status": "error",
        "error": {"type": "FILE_NOT_FOUND", "message": "no such file"},
    })
    warning = gemini_trace_events({
        "type": "error", "severity": "warning", "message": "Loop detected",
    })
    fatal = gemini_trace_events({
        "type": "result", "status": "error",
        "error": {"type": "Error", "message": "quota exhausted"},
    })

    assert [(e.kind, e.message, e.event_id, e.detail) for e in started] == [
        ("tool_start", "Call read_file", "tool-1", "file_path: memory.md"),
    ]
    assert [(e.kind, e.detail) for e in completed] == [("tool_result", "## Names")]
    assert [(e.message, e.detail) for e in failed] == [("Tool error", "no such file")]
    assert [(e.kind, e.message) for e in warning] == [("warning", "Loop detected")]
    assert [(e.kind, e.message) for e in fatal] == [("error", "quota exhausted")]
    # A rejected reply's result names no reason; its error event already did.
    assert gemini_trace_events({"type": "result", "status": "error"}) == []
    assert gemini_trace_events({"type": "result", "status": "success"}) == []
    assert gemini_trace_events({"type": "message", "role": "assistant"}) == []


def test_gemini_backend_assembles_the_streamed_answer(tmp_path: Path) -> None:
    schema = {"type": "object", "required": ["cues"]}
    expected = {"cues": [{"id": "c1", "source": "hello", "translated": "你好"}]}
    seen: dict[str, object] = {}

    class GeminiRunner:
        async def run(self, args, *, on_line=None, **kwargs):
            seen["args"] = list(args)
            seen["stdin"] = kwargs.get("stdin")
            seen["check"] = kwargs.get("check")
            seen["env"] = kwargs["env"]
            seen["cwd"] = cwd = kwargs["cwd"]
            seen["system"] = Path(kwargs["env"]["GEMINI_SYSTEM_MD"]).read_text(encoding="utf-8")
            policy = args[args.index("--policy") + 1]
            seen["policy"] = (cwd / policy).read_text(encoding="utf-8")
            seen["settings"] = json.loads((cwd / ".gemini" / "settings.json").read_text())
            assert on_line is not None
            text = json.dumps(expected, ensure_ascii=False)
            for line in (
                {"type": "init", "model": "flash"},
                {"type": "message", "role": "user", "content": "prompt"},
                {"type": "message", "role": "assistant", "content": "Let me check. ", "delta": True},
                {"type": "tool_use", "tool_name": "glob", "tool_id": "t1", "parameters": {}},
                {"type": "tool_result", "tool_id": "t1", "status": "success"},
                {"type": "message", "role": "assistant", "content": text[:10], "delta": True},
                {"type": "message", "role": "assistant", "content": text[10:], "delta": True},
                {"type": "result", "status": "success", "stats": {}},
            ):
                await on_line("stdout", json.dumps(line, ensure_ascii=False))
            await on_line("stderr", "Attempt 1 failed with status 503")
            return ProcessResult(tuple(str(arg) for arg in args), 0, "", "")

    events = []

    async def exercise():
        return await GeminiBackend(
            tmp_path, GeminiRunner(), ("--approval-mode", "plan")
        ).invoke_with_trace(
            "prompt", system="stable guidance", model="flash", effort="low",
            schema=schema,
            on_event=lambda event: RecordingAwaitable(events, event),
        )

    assert asyncio.run(exercise()) == expected
    args = seen["args"]
    assert args == [
        "gemini", "--prompt", "", "--output-format", "stream-json",
        "--model", "flash", "--policy", "no-tools.toml",
        "--approval-mode", "plan",
    ]
    # The effort picks the working directory, whose settings carry its level.
    assert seen["cwd"] == tmp_path / "agent-gemini-low"
    override = seen["settings"]["modelConfigs"]["customOverrides"][0]
    assert override["match"] == {"model": "chat-base-3", "overrideScope": "core"}
    assert override["modelConfig"]["generateContentConfig"]["thinkingConfig"] == {
        "thinkingLevel": "LOW"
    }
    # Settings load before arguments are parsed, so trust comes from the
    # environment rather than `--skip-trust`.
    env = seen["env"]
    assert isinstance(env, dict)
    assert env["GEMINI_CLI_TRUST_WORKSPACE"] == "true"
    assert seen["stdin"] == b"prompt"
    assert seen["check"] is False
    # Without a schema option, the schema becomes part of the stable guidance.
    system = seen["system"]
    assert isinstance(system, str)
    assert system.startswith("stable guidance\n")
    assert json.dumps(schema) in system
    assert 'toolName = "*"' in seen["policy"] and 'decision = "deny"' in seen["policy"]
    # Commentary before a tool call is shown, never parsed as the answer.
    assert [(event.kind, event.message) for event in events] == [
        ("agent_message", "Let me check."),
        ("tool_start", "Call glob"),
        ("tool_result", "Tool result"),
    ]


def test_gemini_backend_shares_one_directory_and_prompt_file_per_level(
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, str]] = []

    class Runner:
        async def run(self, args, *, on_line=None, **kwargs):
            calls.append((kwargs["cwd"], kwargs["env"]["GEMINI_SYSTEM_MD"]))
            assert on_line is not None
            await on_line("stdout", json.dumps(
                {"type": "message", "role": "assistant", "content": '{"cues": []}'}
            ))
            return ProcessResult(tuple(str(arg) for arg in args), 0, "", "")

    async def exercise() -> None:
        backend = GeminiBackend(tmp_path, Runner())
        for system, effort in (
            ("stage guidance", "high"),
            ("stage guidance", "xhigh"),
            ("other guidance", "high"),
            ("stage guidance", "minimal"),
        ):
            await backend.invoke_with_trace(
                "cues", system=system, model="pro", effort=effort,
                schema={"type": "object"},
            )

    asyncio.run(exercise())

    high = tmp_path / "agent-gemini-high"
    # Every effort gets its own directory; one effort's batches share it.
    assert [cwd for cwd, _ in calls] == [
        high, tmp_path / "agent-gemini-xhigh", high,
        tmp_path / "agent-gemini-minimal",
    ]
    # One prompt file serves every batch with that guidance, whatever the
    # effort; other guidance gets its own.
    assert calls[0][1] == calls[1][1] == calls[3][1] != calls[2][1]
    assert Path(calls[0][1]).parent == tmp_path / "agent-gemini-prompts"
    settings = json.loads((high / ".gemini" / "settings.json").read_text())
    thinking = settings["modelConfigs"]["customOverrides"][0]["modelConfig"][
        "generateContentConfig"
    ]["thinkingConfig"]
    assert thinking == {"thinkingLevel": "HIGH"}
    # Only fixed files sit where Gemini lists the directory ahead of the cues.
    assert sorted(path.name for path in high.iterdir()) == [".gemini", "no-tools.toml"]
    # The effort is passed through as given, even beyond Gemini's own levels.
    xhigh = json.loads(
        (tmp_path / "agent-gemini-xhigh" / ".gemini" / "settings.json").read_text()
    )
    assert xhigh["modelConfigs"]["customOverrides"][0]["modelConfig"][
        "generateContentConfig"
    ]["thinkingConfig"] == {"thinkingLevel": "XHIGH"}


def _gemini_runner(*lines: dict, returncode: int = 0):
    class Runner:
        async def run(self, args, *, on_line=None, **kwargs):
            assert on_line is not None
            for line in lines:
                await on_line("stdout", json.dumps(line, ensure_ascii=False))
            return ProcessResult(tuple(str(arg) for arg in args), returncode, "", "")

    return Runner()


def _assistant(text: str) -> dict:
    return {"type": "message", "role": "assistant", "content": text, "delta": True}


def test_gemini_backend_keeps_the_answer_after_a_silent_retry(tmp_path: Path) -> None:
    expected = {"cues": [{"id": "c1", "source": "hello", "translated": "你好"}]}
    answer = json.dumps(expected, ensure_ascii=False)
    # The first attempt streamed part of a reply before Gemini CLI retried it,
    # and nothing in the output marks where the retry began.
    runner = _gemini_runner(
        _assistant("```json\n" + answer[:25]),
        _assistant("```json\n" + answer[:25]),
        _assistant(answer[25:] + "\n```"),
        {"type": "result", "status": "success"},
    )

    result = asyncio.run(GeminiBackend(tmp_path, runner).invoke_with_trace(
        "prompt", model="flash", effort="low", schema={"type": "object"},
    ))

    assert result == expected


def test_gemini_backend_names_the_reason_for_a_rejected_reply(tmp_path: Path) -> None:
    runner = _gemini_runner(
        _assistant('{"cues": ['),
        {"type": "error", "severity": "error",
         "message": "Response truncated due to token limits."},
        {"type": "result", "status": "error"},
        returncode=1,
    )

    with pytest.raises(
        RuntimeError, match="Gemini CLI failed: Response truncated due to token limits."
    ):
        asyncio.run(GeminiBackend(tmp_path, runner).invoke_with_trace(
            "prompt", model="flash", effort="low", schema={"type": "object"},
        ))


def test_gemini_backend_reports_the_cli_failure_message(tmp_path: Path) -> None:
    class FailingRunner:
        async def run(self, args, *, on_line=None, **kwargs):
            assert on_line is not None
            await on_line("stdout", json.dumps({
                "type": "result", "status": "error",
                "error": {"type": "Error", "message": "No available Gemini accounts"},
            }))
            return ProcessResult(
                tuple(str(arg) for arg in args), 1, "", "    at retryWithBackoff",
            )

    async def exercise():
        await GeminiBackend(tmp_path, FailingRunner()).invoke_with_trace(
            "prompt", model="flash", effort="low", schema={"type": "object"},
        )

    with pytest.raises(RuntimeError, match="Gemini CLI failed: No available Gemini accounts"):
        asyncio.run(exercise())


def test_gemini_backend_raises_the_process_error_without_a_report(tmp_path: Path) -> None:
    class CrashingRunner:
        async def run(self, args, **kwargs):
            return ProcessResult(tuple(str(arg) for arg in args), 42, "", "Unknown argument: policy")

    async def exercise():
        await GeminiBackend(tmp_path, CrashingRunner()).invoke_with_trace(
            "prompt", model="flash", effort="low", schema={"type": "object"},
        )

    with pytest.raises(ProcessError, match="Unknown argument: policy"):
        asyncio.run(exercise())


def test_pipeline_emits_agent_lifecycle_and_readable_result(tmp_path: Path) -> None:
    settings = make_settings(
        target_language="zh-CN", agent={"draft": {"model": "draft"}}
    )
    db = JobDatabase(tmp_path / "db.sqlite3")
    db.upsert_cues([Cue("c1", 0, 1, "user source")])
    events: list[AgentTraceEvent] = []

    def on_agent_event(event: AgentTraceEvent) -> RecordingAwaitable:
        return RecordingAwaitable(events, event)

    pipeline = TranslationPipeline(
        settings, FakeBackend(), db, "", on_agent_event=on_agent_event
    )
    asyncio.run(pipeline.translate_draft(db.list_cues()))

    assert len({event.operation_id for event in events}) == 1
    assert [event.state for event in events if event.kind == "lifecycle"] == [
        "running",
        "completed",
    ]
    assert events[0].kind == "user_message"
    output = next(event for event in events if event.kind == "agent_output")
    assert output.message == "Structured response · 1 cue"
    assert '"source": "agent source"' in (output.detail or "")
    assert '"translated": "代理译文"' in (output.detail or "")
    result = next(event for event in events if event.kind == "result")
    assert events.index(output) < events.index(result)
    assert "updated 1 translations" in result.message
    assert "Translation: 代理译文" in (result.detail or "")
    db.close()


def test_translation_uses_draft_contract(tmp_path: Path) -> None:
    settings = make_settings(
        target_language="zh-CN",
        memory=tmp_path / "memory.md",
        whisper={"model": tmp_path / "m"},
        agent={
            "backend": "codex",
            "draft": {"model": "draft"},
            "final": {"model": "final"},
        },
    )
    db = JobDatabase(tmp_path / "db.sqlite3")
    db.upsert_cues([Cue("c1", 0, 1, "user source")])
    backend = FakeBackend()
    pipeline = TranslationPipeline(settings, backend, db, MEMORY)
    result = asyncio.run(pipeline.translate_draft(db.list_cues()))
    assert result[0].source == "agent source"
    assert result[0].translated == "代理译文"
    system = backend.systems[0]
    assert "target language (zh-CN)" in system
    assert "highly certain ASR/transcription errors" in system
    assert "phonetically very close" in system
    assert "preserve the source text exactly" in system
    assert "do not guess from context" in system
    assert (
        "Follow every applicable terminology, naming, and style constraint in MEMORY"
        in system
    )
    assert "rather than as a word-by-word rendering" in system
    assert "calqued idioms and other translationese" in system
    assert "never add, drop, or embellish content" in system
    # MEMORY reaches the Agent as its own raw-Markdown block rather than as a
    # JSON-escaped string field buried among the task data.
    assert f"<memory>\n{MEMORY.strip()}\n</memory>" in system
    assert "Use MEMORY as the evidence for the source corrections" in system
    assert "plausible misrecognition of something MEMORY records" in system
    # Nothing that repeats across batches may ride along with the cues, or the
    # backends that cache a stable prefix have nothing stable to cache.
    assert backend.prompts[0].startswith("INPUT:\n")
    assert MEMORY.strip() not in backend.prompts[0]
    payload = json.loads(backend.prompts[0].split("INPUT:\n", 1)[1])
    assert "memory" not in payload
    assert set(payload["cues"][0]) == {"id", "source", "translated", "speaker"}
    assert set(backend.schemas[0]["properties"]) == {"cues"}
    assert backend.schemas[0]["title"] == "DraftTranslationResponse"
    item_schema = backend.schemas[0]["properties"]["cues"]["items"]
    assert set(item_schema["properties"]) == {"id", "source", "translated"}
    db.close()


def test_draft_batches_see_the_cues_on_both_sides(tmp_path: Path) -> None:
    settings = make_settings(
        target_language="zh-CN",
        agent={
            "backend": "codex",
            "draft": {
                "model": "draft",
                "batch_size": 1,
                "preceding_context": 1,
                "following_context": 2,
            },
        },
    )
    db = JobDatabase(tmp_path / "db.sqlite3")
    cues = [Cue(str(index), index, index + 1, f"s{index}") for index in range(1, 5)]
    db.upsert_cues(cues)
    backend = EchoBackend()
    pipeline = TranslationPipeline(settings, backend, db, "")

    asyncio.run(pipeline.translate_draft(db.list_cues()))

    payloads = {
        json.loads(prompt.split("INPUT:\n", 1)[1])["cues"][0]["id"]:
            json.loads(prompt.split("INPUT:\n", 1)[1])
        for prompt in backend.prompts
    }
    assert [cue["id"] for cue in payloads["1"]["preceding_context"]] == []
    assert [cue["id"] for cue in payloads["1"]["following_context"]] == ["2", "3"]
    assert [cue["id"] for cue in payloads["3"]["preceding_context"]] == ["2"]
    assert [cue["id"] for cue in payloads["3"]["following_context"]] == ["4"]
    assert [cue["id"] for cue in payloads["4"]["following_context"]] == []
    assert "never translate them and never return them" in backend.systems[0]
    assert "translate only the part that belongs to `cues`" in backend.systems[0]
    # Every batch of a stage must see the identical system text, or a cached
    # prefix would be rewritten on each one.
    assert len(set(backend.systems)) == 1
    db.close()


def test_streaming_batches_have_no_following_context(tmp_path: Path) -> None:
    # Nothing after the newest cue has been transcribed yet.
    settings = make_settings(
        target_language="zh-CN",
        agent={
            "backend": "codex",
            "draft": {"model": "draft", "batch_size": 2, "following_context": 5},
        },
    )
    db = JobDatabase(tmp_path / "db.sqlite3")
    batch = [Cue("3", 2, 3, "s3"), Cue("4", 3, 4, "s4")]
    db.upsert_cues([Cue("1", 0, 1, "s1"), Cue("2", 1, 2, "s2"), *batch])
    backend = EchoBackend()
    pipeline = TranslationPipeline(settings, backend, db, "")

    asyncio.run(
        pipeline.translate_draft(
            batch, preceding_context=[Cue("2", 1, 2, "s2")]
        )
    )

    payload = json.loads(backend.prompts[0].split("INPUT:\n", 1)[1])
    assert [cue["id"] for cue in payload["preceding_context"]] == ["2"]
    assert payload["following_context"] == []
    db.close()


def test_post_alignment_translation_ignores_agent_source_edits(tmp_path: Path) -> None:
    settings = make_settings(
        target_language="zh-CN",
        agent={"backend": "codex", "draft": {"model": "draft"}},
    )
    db = JobDatabase(tmp_path / "db.sqlite3")
    db.upsert_cues([Cue("c1", 0, 1, "forced-aligned sentence")])
    backend = FakeBackend()
    pipeline = TranslationPipeline(settings, backend, db, "")

    result = asyncio.run(
        pipeline.translate_draft(db.list_cues(), translate_only=True)
    )

    assert result[0].source == "forced-aligned sentence"
    assert result[0].translated == "代理译文"
    assert "do not modify, correct, merge, or split the source text" in backend.systems[0]
    db.close()


def test_draft_prompt_omits_memory_rules_without_memory(tmp_path: Path) -> None:
    settings = make_settings(
        target_language="zh-CN",
        agent={"backend": "codex", "draft": {"model": "draft"}},
    )
    db = JobDatabase(tmp_path / "db.sqlite3")
    db.upsert_cues([Cue("c1", 0, 1, "user source")])
    backend = FakeBackend()
    pipeline = TranslationPipeline(settings, backend, db, "   \n")

    asyncio.run(pipeline.translate_draft(db.list_cues()))

    # A rule pointing at guidance that was never supplied is noise the Agent
    # has to resolve on its own.
    assert "MEMORY" not in backend.systems[0]
    assert "highly certain ASR/transcription errors" in backend.systems[0]
    db.close()


def test_post_alignment_memory_rule_keeps_the_source_frozen(tmp_path: Path) -> None:
    settings = make_settings(
        target_language="zh-CN",
        agent={"backend": "codex", "draft": {"model": "draft"}},
    )
    db = JobDatabase(tmp_path / "db.sqlite3")
    db.upsert_cues([Cue("c1", 0, 1, "forced-aligned sentence")])
    backend = FakeBackend()
    pipeline = TranslationPipeline(settings, backend, db, MEMORY)

    asyncio.run(pipeline.translate_draft(db.list_cues(), translate_only=True))

    system = backend.systems[0]
    assert f"<memory>\n{MEMORY.strip()}\n</memory>" in system
    assert "leave the source text unchanged even where it contradicts MEMORY" in system
    assert "Use MEMORY as the evidence for the source corrections" not in system
    db.close()


def test_empty_translation_is_retried_and_reported(tmp_path: Path) -> None:
    settings = make_settings(
        target_language="zh-CN",
        agent={
            "draft": {"model": "draft", "max_attempts": 3, "retry_delay_seconds": 0},
        },
    )
    db = JobDatabase(tmp_path / "db.sqlite3")
    db.upsert_cues([Cue("c1", 0, 1, "hello")])
    backend = SequenceBackend(["", "你好"])
    retries: list[str] = []

    async def on_retry(message: str) -> None:
        retries.append(message)

    pipeline = TranslationPipeline(settings, backend, db, "", on_retry=on_retry)
    result = asyncio.run(pipeline.translate_draft(db.list_cues()))

    assert backend.calls == 2
    assert result[0].translated == "你好"
    assert len(retries) == 1
    assert "empty translations" in retries[0]
    assert "retrying 2/3" in retries[0]
    statuses = [
        row["status"]
        for row in db.connection.execute(
            "SELECT status FROM translation_batches ORDER BY id"
        )
    ]
    assert statuses == ["failed", "complete"]
    db.close()


def test_timed_out_agent_batch_is_cancelled_and_retried(tmp_path: Path) -> None:
    settings = make_settings(
        target_language="zh-CN",
        agent={
            "draft": {
                "model": "draft",
                "timeout_seconds": 0.01,
                "max_attempts": 2,
                "retry_delay_seconds": 0,
            },
        },
    )
    db = JobDatabase(tmp_path / "db.sqlite3")
    db.upsert_cues([Cue("c1", 0, 1, "hello")])
    backend = TimeoutThenSuccessBackend()
    retries: list[str] = []

    async def on_retry(message: str) -> None:
        retries.append(message)

    pipeline = TranslationPipeline(settings, backend, db, "", on_retry=on_retry)
    result = asyncio.run(pipeline.translate_draft(db.list_cues()))

    assert backend.calls == 2
    assert result[0].translated == "你好"
    assert "timed out after 0.01s" in retries[0]
    db.close()


def test_empty_translation_fails_after_retry_limit(tmp_path: Path) -> None:
    settings = make_settings(
        target_language="zh-CN",
        agent={
            "draft": {"model": "draft", "max_attempts": 2, "retry_delay_seconds": 0},
        },
    )
    db = JobDatabase(tmp_path / "db.sqlite3")
    db.upsert_cues([Cue("c1", 0, 1, "hello")])
    backend = SequenceBackend([""])
    pipeline = TranslationPipeline(settings, backend, db, "")

    with pytest.raises(ValueError, match="empty translations"):
        asyncio.run(pipeline.translate_draft(db.list_cues()))

    assert backend.calls == 2
    assert db.list_cues()[0].translated is None
    assert db.connection.execute(
        "SELECT COUNT(*) FROM translation_batches WHERE status = 'failed'"
    ).fetchone()[0] == 2
    db.close()
