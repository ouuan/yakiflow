import asyncio

import pytest

pytest.importorskip("textual")

from textual.app import App, ComposeResult
from textual.widgets import DataTable, RichLog

from conftest import make_settings
from yakiflow import tui
from yakiflow.database import JobDatabase
from yakiflow.job import YakiFlowJob
from yakiflow.memory import MemoryStore
from yakiflow.models import AgentTraceEvent, Cue, JobEvent
from yakiflow.subtitle_preview import SubtitlePreviewApp
from yakiflow.subtitles import render_ass
from yakiflow.tui import (
    AgentConversationLog,
    AgentInspector,
    AlignmentModelFailureDialog,
    PipelineProgressBar,
    SubtitleTable,
    YakiFlowApp,
)


def _subtitle(cues: list[str]) -> str:
    return render_ass(
        [
            Cue(str(index + 1), float(index), index + 0.9, text)
            for index, text in enumerate(cues)
        ],
        "source",
    )


def test_pipeline_progress_bar_keeps_explicit_eta_on_refresh() -> None:
    class ProgressApp(App[None]):
        def compose(self) -> ComposeResult:
            yield PipelineProgressBar(total=100, id="progress")

    async def exercise() -> None:
        app = ProgressApp()
        async with app.run_test():
            progress = app.query_one("#progress", PipelineProgressBar)
            progress.update(progress=25)
            progress.set_estimated_remaining(123)
            progress.update()
            assert progress._display_eta == 123

    asyncio.run(exercise())


def test_pipeline_failure_renders_traceback_in_conversation(tmp_path) -> None:
    class FailingJob:
        def __init__(self) -> None:
            self.db = JobDatabase(tmp_path / "job.sqlite3")
            self.work_dir = tmp_path
            self.listener = None
            self.diarization_enabled = False

        async def run(self):
            raise RuntimeError("timing adjustment exploded")

    async def exercise() -> None:
        job = FailingJob()
        app = YakiFlowApp(job)
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause()
            await pilot.pause()
            conversation = app.query_one("#conversation", RichLog)
            rendered = "\n".join(line.text for line in conversation.lines)
            assert "Failed: timing adjustment exploded" in rendered
            traceback_path = tmp_path / "yakiflow-traceback.log"
            assert traceback_path.is_file()
            assert "RuntimeError: timing adjustment exploded" in traceback_path.read_text()
            assert "Full traceback saved to:" in rendered
        job.db.close()

    asyncio.run(exercise())


def test_alignment_model_failure_dialog_requires_retry_or_fallback(tmp_path) -> None:
    class DecisionJob:
        def __init__(self) -> None:
            self.db = JobDatabase(tmp_path / "job.sqlite3")
            self.work_dir = tmp_path
            self.listener = None
            self.diarization_enabled = False
            self.alignment_model_failure_listener = None
            self.decisions: list[str] = []

        async def run(self):
            decision = await self.alignment_model_failure_listener(
                "WhisperX model download timed out"
            )
            self.decisions.append(decision)
            if decision == "retry":
                self.decisions.append(
                    await self.alignment_model_failure_listener(
                        "WhisperX model download failed again"
                    )
                )
            return []

    async def exercise() -> None:
        job = DecisionJob()
        app = YakiFlowApp(job)
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, AlignmentModelFailureDialog)
            assert app.screen.query_one("#alignment-retry")
            assert app.screen.query_one("#alignment-fallback")

            await pilot.click("#alignment-retry")
            await pilot.pause()
            assert isinstance(app.screen, AlignmentModelFailureDialog)
            await pilot.click("#alignment-fallback")
            await pilot.pause()
            assert job.decisions == ["retry", "fallback"]
            assert app.succeeded
        job.db.close()

    asyncio.run(exercise())


def test_speaker_column_exists_only_for_diarized_jobs() -> None:
    class TableApp(App[None]):
        def compose(self) -> ComposeResult:
            yield SubtitleTable(id="plain")
            yield SubtitleTable(id="diarized", show_speaker=True)

    async def exercise() -> None:
        app = TableApp()
        async with app.run_test(size=(80, 20)) as pilot:
            plain = app.query_one("#plain", SubtitleTable)
            diarized = app.query_one("#diarized", SubtitleTable)
            assert [str(column.label) for column in plain.columns.values()] == [
                "No.", "Start", "Source", "Translation",
            ]
            assert [str(column.label) for column in diarized.columns.values()] == [
                "No.", "Start", "Speaker", "Source", "Translation",
            ]

            cue = Cue("1", 0.0, 1.0, "hello", speaker="Alice")
            plain.add_subtitle(cue)
            diarized.add_subtitle(cue)
            await pilot.pause()
            assert diarized.get_cell("1", "speaker").plain.strip() == "Alice"

            diarized.update_subtitle(
                Cue("1", 0.0, 1.0, "hello", speaker="Bob")
            )
            plain.update_subtitle(cue)
            await pilot.pause()
            assert diarized.get_cell("1", "speaker").plain.strip() == "Bob"

    asyncio.run(exercise())


def test_subtitle_tail_follow_tracks_user_intent() -> None:
    class TableApp(App[None]):
        CSS = "#recent { height: 10; overflow-y: scroll; }"

        def compose(self) -> ComposeResult:
            yield SubtitleTable(id="recent")

    async def exercise() -> None:
        app = TableApp()
        async with app.run_test(size=(80, 20)) as pilot:
            table = app.query_one("#recent", SubtitleTable)
            for index in range(30):
                table.add_subtitle(Cue(str(index), index, index + 1, f"cue {index}"))
            await pilot.pause()
            assert all(row.height >= 2 for row in table.rows.values())
            table.user_scroll_end()
            assert table.follow_tail

            bottom = table.scroll_target_y
            half_page = max(1, table.scrollable_content_region.height // 2)
            table.user_scroll_page_up()
            assert table.scroll_target_y == bottom - half_page
            assert not table.follow_tail
            table.user_scroll_page_down()
            assert table.scroll_target_y == bottom
            await pilot.pause()
            assert table.follow_tail

            table.user_scroll_up()
            assert not table.follow_tail
            held_position = table.scroll_target_y
            table.add_subtitle(Cue("30", 30, 31, "cue 30"))
            table.call_after_refresh(
                table.scroll_to, y=held_position, animate=False, immediate=True
            )
            await pilot.pause()
            assert table.scroll_target_y == held_position

            table.user_scroll_end()
            table.add_subtitle(Cue("31", 31, 32, "cue 31"))
            table.call_after_refresh(
                table.scroll_end, animate=False, force=True, immediate=True
            )
            await pilot.pause()
            assert table.follow_tail
            assert table.is_vertical_scroll_end

    asyncio.run(exercise())


def test_live_subtitle_reload_preserves_user_scroll_position(tmp_path) -> None:
    subtitle = tmp_path / "review.ass"
    marker = tmp_path / "done"
    cues = [f"cue {index}" for index in range(40)]
    subtitle.write_text(_subtitle(cues), encoding="utf-8")

    async def exercise() -> None:
        app = SubtitlePreviewApp(subtitle, marker)
        async with app.run_test(size=(80, 20)) as pilot:
            await pilot.pause()
            table = app.query_one("#recent", SubtitleTable)
            table.user_scroll_end()
            table.user_scroll_page_up()
            await pilot.pause()
            held_position = table.scroll_y
            assert not table.follow_tail

            cues[20] = "updated cue 20 with a longer Agent correction"
            subtitle.write_text(_subtitle(cues), encoding="utf-8")
            app._refresh()
            await pilot.pause()

            assert table.scroll_y == held_position
            assert not table.follow_tail

    asyncio.run(exercise())


def test_live_preview_shows_speakers_only_for_diarized_jobs(tmp_path) -> None:
    subtitle = tmp_path / "review.ass"
    marker = tmp_path / "done"
    subtitle.write_text(
        render_ass(
            [
                Cue("1", 0.0, 1.0, "hello", speaker="Alice"),
                Cue("2", 1.0, 2.0, "hi", speaker="Bob"),
            ],
            "source",
        ),
        encoding="utf-8",
    )

    async def exercise() -> None:
        diarized = SubtitlePreviewApp(subtitle, marker, show_speaker=True)
        async with diarized.run_test(size=(80, 20)) as pilot:
            await pilot.pause()
            table = diarized.query_one("#recent", SubtitleTable)
            assert table.get_cell("1", "speaker").plain.strip() == "Alice"
            assert table.get_cell("2", "speaker").plain.strip() == "Bob"

        plain = SubtitlePreviewApp(subtitle, marker)
        async with plain.run_test(size=(80, 20)) as pilot:
            await pilot.pause()
            table = plain.query_one("#recent", SubtitleTable)
            assert [str(column.label) for column in table.columns.values()] == [
                "No.", "Start", "Source", "Translation",
            ]

    asyncio.run(exercise())


def test_successful_pipeline_exits_for_interactive_agent_handoff(tmp_path) -> None:
    class FakeJob:
        def __init__(self) -> None:
            self.db = JobDatabase(tmp_path / "job.sqlite3")
            self.memory_store = MemoryStore(tmp_path / "memory.md")
            self.settings = None
            self.backend = None
            self.listener = None
            self.diarization_enabled = False
            self.work_dir = tmp_path

        async def run(self):
            return []

        def finish_review(self) -> None:
            pass

    async def exercise() -> None:
        job = FakeJob()
        app = YakiFlowApp(job)
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause()
            assert app.succeeded
        job.db.close()

    asyncio.run(exercise())


def test_stop_shortcut_cancels_pipeline_without_immediately_exiting_tui(
    tmp_path,
) -> None:
    class StoppableJob:
        def __init__(self) -> None:
            self.db = JobDatabase(tmp_path / "job.sqlite3")
            self.work_dir = tmp_path
            self.listener = None
            self.diarization_enabled = False
            self.alignment_model_failure_listener = None
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def run(self):
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    async def exercise() -> None:
        job = StoppableJob()
        app = YakiFlowApp(job)
        async with app.run_test(size=(80, 30)) as pilot:
            await job.started.wait()
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert not job.cancelled.is_set()

            await pilot.press("s")
            await asyncio.wait_for(job.cancelled.wait(), timeout=1)
            await pilot.pause()

            assert app.is_running
            assert not app.succeeded
            conversation = app.query_one("#conversation", RichLog)
            rendered = "\n".join(line.text for line in conversation.lines)
            assert "Stop requested" in rendered
            assert "Interrupted; resume" in rendered
            assert str(tmp_path) in rendered

            await pilot.press("ctrl+q")
        job.db.close()

    asyncio.run(exercise())


def test_open_media_shortcut_opens_growing_stream_media(tmp_path, monkeypatch) -> None:
    job = YakiFlowJob(
        "https://example.test/live",
        make_settings(
            agent={"backend": "codex"},
            work_dir=tmp_path / "work",
            memory=tmp_path / "memory.md",
        ),
        backend=object(),
    )
    growing = tmp_path / "downloads" / "source-abc.mkv.part"
    growing.parent.mkdir()
    growing.write_bytes(b"media")
    partial = job.work_dir / "source.live.incomplete.ass"
    partial.write_text(_subtitle(["hello"]), encoding="utf-8")

    started = asyncio.Event()

    async def fake_run():
        # What _media_stage does once the stream's download decodes.
        job._stream_media_path = growing
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(job, "run", fake_run)
    calls: list[tuple] = []

    def fake_open_media(command, media_path, subtitle_path, *, cwd=None):
        calls.append((command, media_path, subtitle_path, cwd))
        return True

    monkeypatch.setattr(tui, "open_media", fake_open_media)

    async def exercise() -> None:
        app = YakiFlowApp(job)
        async with app.run_test(size=(80, 30)) as pilot:
            await asyncio.wait_for(started.wait(), timeout=1)
            await pilot.press("o")
            await pilot.pause()
            assert calls == [
                (
                    job.settings.review.video_open_command,
                    growing,
                    partial,
                    job.work_dir,
                )
            ]

    asyncio.run(exercise())
    job.close()


def test_resumed_app_loads_all_existing_subtitles(tmp_path) -> None:
    class FakeJob:
        def __init__(self) -> None:
            self.db = JobDatabase(tmp_path / "job.sqlite3")
            self.db.upsert_cues([
                Cue("1", 0, 1, "first", "第一"),
                Cue("2", 1, 2, "second"),
            ])
            self.memory_store = MemoryStore(tmp_path / "memory.md")
            self.settings = None
            self.backend = None
            self.listener = None
            self.diarization_enabled = False
            self.work_dir = tmp_path

        async def run(self):
            await self.listener(
                JobEvent("transcript", "Whisper subtitle", cue=Cue("3", 2, 3, "third"))
            )
            return []

        def finish_review(self) -> None:
            pass

    async def exercise() -> None:
        job = FakeJob()
        app = YakiFlowApp(job)
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause()
            table = app.query_one("#recent", SubtitleTable)
            assert list(table.rows) == ["1", "2", "3"]
            assert table.get_cell("1", "source").plain.rstrip() == "first"
            assert table.get_cell("1", "translation").plain.rstrip() == "第一"
            assert table.get_cell("2", "source").plain.rstrip() == "second"
            assert table.get_cell("3", "source").plain.rstrip() == "third"
            assert app.whisper_cues == 3
            assert app.processed_ids == {"1"}
        job.db.close()

    asyncio.run(exercise())


def test_timeline_replaced_event_reloads_and_renumbers_rows(tmp_path) -> None:
    class FakeJob:
        def __init__(self) -> None:
            self.db = JobDatabase(tmp_path / "job.sqlite3")
            self.db.upsert_cues([Cue("old", 0, 1, "old")])
            self.memory_store = MemoryStore(tmp_path / "memory.md")
            self.settings = None
            self.backend = None
            self.listener = None
            self.diarization_enabled = False
            self.work_dir = tmp_path

        async def run(self):
            self.db.replace_aligned_timeline([
                Cue("1", 0.1, 0.6, "first sentence", "第一句"),
                Cue("2", 0.6, 1.2, "second sentence"),
            ])
            await self.listener(JobEvent("timeline-replaced", "replaced"))
            return []

        def finish_review(self) -> None:
            pass

    async def exercise() -> None:
        job = FakeJob()
        app = YakiFlowApp(job)
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause()
            table = app.query_one("#recent", SubtitleTable)
            assert list(table.rows) == ["1", "2"]
            assert table.get_cell("1", "source").plain.rstrip() == "first sentence"
            assert app.whisper_cues == 2
            assert app.processed_ids == {"1"}
        job.db.close()

    asyncio.run(exercise())


def test_resumed_app_sorts_mid_draft_word_rows_by_time(tmp_path) -> None:
    class FakeJob:
        def __init__(self) -> None:
            self.db = JobDatabase(tmp_path / "job.sqlite3")
            # Batches land in completion order, not time order.
            self.db.upsert_cues([
                Cue("w40-59", 8.0, 11.0, "tail", "尾"),
                Cue("w0-19", 0.0, 3.0, "head", "头"),
            ])
            self.memory_store = MemoryStore(tmp_path / "memory.md")
            self.settings = None
            self.backend = None
            self.listener = None
            self.diarization_enabled = False
            self.work_dir = tmp_path

        async def run(self):
            return []

        def finish_review(self) -> None:
            pass

    async def exercise() -> None:
        job = FakeJob()
        app = YakiFlowApp(job)
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause()
            table = app.query_one("#recent", SubtitleTable)
            assert list(table.rows) == ["w0-19", "w40-59"]
        job.db.close()

    asyncio.run(exercise())


def test_timeline_replaced_event_with_cues_uses_the_payload_view(tmp_path) -> None:
    class FakeJob:
        def __init__(self) -> None:
            self.db = JobDatabase(tmp_path / "job.sqlite3")
            # A stale durable row proves the payload, not the database, wins.
            self.db.upsert_cues([Cue("stale", 0, 1, "stale")])
            self.memory_store = MemoryStore(tmp_path / "memory.md")
            self.settings = None
            self.backend = None
            self.listener = None
            self.diarization_enabled = False
            self.work_dir = tmp_path

        async def run(self):
            await self.listener(JobEvent(
                "timeline-replaced",
                "partial draft timeline updated",
                cues=[
                    Cue("w0-2", 0.1, 0.6, "first words", "第一句"),
                    Cue("preview-1", 0.6, 1.2, "still transcribing"),
                ],
            ))
            return []

        def finish_review(self) -> None:
            pass

    async def exercise() -> None:
        job = FakeJob()
        app = YakiFlowApp(job)
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause()
            table = app.query_one("#recent", SubtitleTable)
            assert list(table.rows) == ["w0-2", "preview-1"]
            assert table.get_cell("w0-2", "translation").plain.rstrip() == "第一句"
            assert app.whisper_cues == 2
            assert app.processed_ids == {"w0-2"}
        job.db.close()

    asyncio.run(exercise())


def test_f2_agent_inspector_shows_conversation_and_tool_calls(tmp_path) -> None:
    class FakeJob:
        def __init__(self) -> None:
            self.db = JobDatabase(tmp_path / "job.sqlite3")
            self.memory_store = MemoryStore(tmp_path / "memory.md")
            self.settings = None
            self.backend = None
            self.listener = None
            self.diarization_enabled = False
            self.work_dir = tmp_path

        async def run(self):
            await asyncio.Event().wait()

        def finish_review(self) -> None:
            pass

    async def exercise() -> None:
        job = FakeJob()
        app = YakiFlowApp(job)
        async with app.run_test(size=(100, 38)) as pilot:
            await pilot.pause()
            for trace in (
                AgentTraceEvent(
                    "operation-1", "user_message",
                    "Draft translate cues 1–20", cue_ids=("1", "20"), model="test-model",
                ),
                AgentTraceEvent(
                    "operation-1", "lifecycle",
                    "Draft translation", cue_ids=("1", "20"), model="test-model",
                ),
                AgentTraceEvent(
                    "operation-1", "tool_start", "Run rg terminology",
                    cue_ids=("1", "20"), detail="cwd: /work",
                ),
                AgentTraceEvent(
                    "operation-1", "tool_result", "Run rg terminology · exit 0",
                    cue_ids=("1", "20"), detail="4 matches",
                ),
                AgentTraceEvent(
                    "operation-1",
                    "agent_output",
                    "Structured response · 1 cue",
                    cue_ids=("1", "20"),
                    detail=(
                        '{\n  "cues": [{\n    "id": "1",\n'
                        '    "source": "今日は晴れです。",\n'
                        '    "translated": "It is sunny today."\n  }]\n}'
                    ),
                ),
            ):
                await app._event(JobEvent("agent_trace", trace.message, agent_trace=trace))
            for index in range(30):
                trace = AgentTraceEvent(
                    "operation-1", "agent_message",
                    f"Conversation line {index}", cue_ids=("1", "20"),
                )
                await app._event(JobEvent("agent_trace", trace.message, agent_trace=trace))
            second = AgentTraceEvent(
                "operation-2", "user_message",
                "Draft translate cues 21–40", cue_ids=("21", "40"),
            )
            await app._event(JobEvent("agent_trace", second.message, agent_trace=second))
            for index in range(30):
                trace = AgentTraceEvent(
                    "operation-2", "agent_message",
                    f"Second conversation line {index}", cue_ids=("21", "40"),
                )
                await app._event(JobEvent("agent_trace", trace.message, agent_trace=trace))
            await pilot.press("f2")
            await pilot.pause()

            assert isinstance(app.screen, AgentInspector)
            table = app.screen.query_one("#agent-list", DataTable)
            assert table.row_count == 2
            detail = app.screen.query_one("#agent-detail", AgentConversationLog)
            rendered = "\n".join(line.text for line in detail.lines)
            assert "You" in rendered
            assert "Run rg terminology" in rendered
            assert "4 matches" in rendered
            assert "Agent output · Structured response · 1 cue" in rendered
            assert '"source": "今日は晴れです。"' in rendered
            assert '"translated": "It is sunny today."' in rendered

            detail.user_scroll_end()
            await pilot.press("pageup")
            held_position = detail.scroll_target_y
            assert held_position < detail.max_scroll_y
            assert not detail.follow_tail

            for index in range(3):
                appended = AgentTraceEvent(
                    "operation-1", "agent_message",
                    f"New output {index} while reviewing older content",
                    cue_ids=("1", "20"),
                )
                await app._event(
                    JobEvent("agent_trace", appended.message, agent_trace=appended)
                )
                await pilot.pause()
                assert detail.scroll_target_y == held_position
                assert not detail.follow_tail

            # The inspector refreshes elapsed times once per second. That
            # periodic table refresh must not rebuild or move the conversation.
            await pilot.pause(1.1)
            assert detail.scroll_target_y == held_position
            assert not detail.follow_tail

            await pilot.press("down")
            await pilot.pause()
            assert app.screen.selected_operation == "operation-2"

            detail.user_scroll_end()
            await pilot.press("pageup")
            second_position = detail.scroll_target_y
            assert second_position < detail.max_scroll_y
            await pilot.pause(1.1)
            assert app.screen.selected_operation == "operation-2"
            assert detail.scroll_target_y == second_position
            assert not detail.follow_tail

            await pilot.press("f2")
            await pilot.pause()
            assert not isinstance(app.screen, AgentInspector)
        job.db.close()

    asyncio.run(exercise())


def test_stream_completion_dialog_disables_review_without_results():
    from textual.widgets import Button
    from yakiflow.tui import StreamCompletionDialog
    class DialogApp(App):
        def on_mount(self):
            self.push_screen(StreamCompletionDialog(False))
    async def exercise():
        app = DialogApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.screen.query_one("#review", Button).disabled
            assert not app.screen.query_one("#retranscribe", Button).disabled
            assert not app.screen.query_one("#exit", Button).disabled
    asyncio.run(exercise())
