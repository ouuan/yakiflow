from __future__ import annotations

import asyncio
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .alignment import AlignmentModelFailureDecision
from .job import StreamExit, YakiFlowJob
from .media_player import open_media
from .models import (
    AgentTraceEvent,
    Cue,
    JobEvent,
    is_preview_cue_id,
    is_word_cue_id,
)

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.worker import Worker
from textual.widgets import Button, DataTable, Footer, Header, ProgressBar, RichLog, Static


def format_start_time(seconds: float) -> str:
    """Render a cue start that always fits the fixed-width Start column.

    Past an hour the centiseconds are dropped rather than the low digits: an
    ``MM:SS.CC`` value would overflow the column and be cropped, so two cues a
    minute apart could render identically.
    """
    centiseconds = max(0, round(seconds * 100))
    minutes, remainder = divmod(centiseconds, 6_000)
    secs, fraction = divmod(remainder, 100)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}.{fraction:02d}"


def subtitle_ordinal(cue: Cue) -> str:
    """Render the No. column, blank while the cue split is still provisional.

    A preview cue is renumbered on every re-segmentation, and a word-range
    agent cue can still be replaced by a junction repair, so showing either
    ID would only offer a sequence that keeps shifting under the reader.
    Once the pipeline settles a cue's final number it renames the cue in
    place, and the numeric ID shows here mid-draft.
    """
    if is_preview_cue_id(cue.id) or is_word_cue_id(cue.id):
        return ""
    return str(cue.id)


def subtitle_text_widths(
    viewport_width: int,
    *,
    start_width: int = 8,
    ordinal_width: int = 5,
    speaker_width: int = 0,
    cell_padding: int = 1,
) -> tuple[int, int]:
    """Split the actual table viewport without creating horizontal overflow.

    ``speaker_width`` is 0 when the table has no Speaker column; the column's
    two cell paddings are then not rendered either, so they are counted only
    alongside a real width.
    """
    fixed_render_width = start_width + ordinal_width + 4 * cell_padding
    if speaker_width:
        fixed_render_width += speaker_width + 2 * cell_padding
    text_column_padding = 4 * cell_padding
    available = max(2, viewport_width - fixed_render_width - text_column_padding)
    source_width = available // 2
    return source_width, available - source_width


@dataclass(slots=True)
class AgentTaskState:
    operation_id: str
    display_id: str
    state: str = "running"
    cue_ids: tuple[str, ...] = ()
    model: str | None = None
    attempt: int | None = None
    max_attempts: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    events: list[AgentTraceEvent] = field(default_factory=list)

    def apply(self, event: AgentTraceEvent) -> None:
        self.state = event.state
        if event.cue_ids:
            self.cue_ids = event.cue_ids
        if event.model:
            self.model = event.model
        if event.attempt is not None:
            self.attempt = event.attempt
        if event.max_attempts is not None:
            self.max_attempts = event.max_attempts
        self.started_at = self.started_at or event.created_at
        if event.state in {"completed", "failed", "cancelled"}:
            self.finished_at = event.created_at
        self.events.append(event)

    @property
    def active(self) -> bool:
        return self.state in {"running", "retrying"}

    @property
    def task_label(self) -> str:
        label = "Draft translation"
        if self.cue_ids:
            span = self.cue_ids[0] if len(self.cue_ids) == 1 else f"{self.cue_ids[0]}–{self.cue_ids[-1]}"
            label += f" · cues {span}"
        elif self.events:
            # Word-batch agents have no cue IDs before they run; their opening
            # summary ("Segment and translate words 12–411") is the task name.
            label = self.events[0].message
        return label

    @property
    def attempt_label(self) -> str:
        if self.attempt is None:
            return "—"
        return f"{self.attempt}/{self.max_attempts}" if self.max_attempts else str(self.attempt)

    def elapsed_seconds(self) -> int:
        if self.started_at is None:
            return 0
        try:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.finished_at) if self.finished_at else datetime.now(UTC)
        except ValueError:
            return 0
        return max(0, round((end - start).total_seconds()))

    @property
    def elapsed_label(self) -> str:
        minutes, seconds = divmod(self.elapsed_seconds(), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def subtitle_cell(text: str, *, no_wrap: bool = False) -> Text:
    """Render one trailing blank line as inter-cue vertical padding."""
    return Text(
        text.rstrip("\n") + "\n",
        overflow="fold",
        no_wrap=no_wrap,
    )


class ReadOnlyRichLog(RichLog, can_focus=True):
    """Scrollable output pane that can be focused for traceback inspection."""


class PipelineProgressBar(ProgressBar):
    """A progress bar whose ETA is supplied by the pipeline estimator."""

    def __init__(self, *args, **kwargs):
        self._pipeline_eta: int | None = None
        super().__init__(*args, **kwargs)

    def update(self, **kwargs) -> None:
        super().update(**kwargs)
        # Textual refreshes its generic ETA once per second. Restore the
        # stage-aware value after both explicit and timer-driven updates.
        self._display_eta = self._pipeline_eta

    def set_estimated_remaining(self, seconds: int | None) -> None:
        self._pipeline_eta = seconds
        self._display_eta = seconds


class FollowTailMixin:
    """Preserve whether the user wants a live scroll view to follow its tail."""

    follow_tail: bool

    def _init_follow_tail(self) -> None:
        self.follow_tail = True

    def _detach_follow_tail(self) -> None:
        if self.max_scroll_y > 0 and self.scroll_target_y > 0:
            self.follow_tail = False

    def _resume_follow_tail_if_at_end(self) -> None:
        if self.scroll_target_y >= self.max_scroll_y - 1:
            self.follow_tail = True

    def restore_scroll_after_update(self, previous_y: float) -> None:
        if self.follow_tail:
            self.call_after_refresh(
                self.scroll_end, animate=False, force=True, immediate=True
            )
        else:
            self.call_after_refresh(
                self.scroll_to,
                y=previous_y,
                animate=False,
                immediate=True,
            )

    def user_scroll_up(self) -> None:
        self._detach_follow_tail()
        self.scroll_up(animate=False)

    def user_scroll_down(self) -> None:
        self.scroll_down(animate=False)
        self.call_after_refresh(self._resume_follow_tail_if_at_end)

    def user_scroll_page_up(self) -> None:
        self._detach_follow_tail()
        distance = max(1, self.scrollable_content_region.height // 2)
        self.scroll_to(
            y=self.scroll_target_y - distance,
            animate=False,
            immediate=True,
        )

    def user_scroll_page_down(self) -> None:
        distance = max(1, self.scrollable_content_region.height // 2)
        self.scroll_to(
            y=self.scroll_target_y + distance,
            animate=False,
            immediate=True,
        )
        self.call_after_refresh(self._resume_follow_tail_if_at_end)

    def user_scroll_home(self) -> None:
        self._detach_follow_tail()
        self.scroll_home(animate=False, force=True, immediate=True)

    def user_scroll_end(self) -> None:
        self.follow_tail = True
        self.scroll_end(animate=False, force=True, immediate=True)

    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        if not event.ctrl and not event.shift:
            self._detach_follow_tail()
        super()._on_mouse_scroll_up(event)

    def _on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        super()._on_mouse_scroll_down(event)
        if not event.ctrl and not event.shift:
            self.call_after_refresh(self._resume_follow_tail_if_at_end)

    def _on_scroll_up(self, event) -> None:
        self._detach_follow_tail()
        super()._on_scroll_up(event)

    def _on_scroll_down(self, event) -> None:
        super()._on_scroll_down(event)
        self.call_after_refresh(self._resume_follow_tail_if_at_end)

    def _on_scroll_to(self, message) -> None:
        if message.y is not None:
            self.follow_tail = message.y >= self.max_scroll_y - 1
        super()._on_scroll_to(message)


class SubtitleTable(FollowTailMixin, DataTable, can_focus=True):
    """Virtualized subtitle rows, keyed by the stable Whisper cue ID."""

    def __init__(self, *, show_speaker: bool = False, **kwargs):
        super().__init__(
            show_cursor=False, cursor_type="none", zebra_stripes=True,
            cell_padding=1, **kwargs,
        )
        # Only present when the backend diarizes: every other run would show a
        # permanently empty column and waste subtitle width on it.
        self.show_speaker = show_speaker
        self.add_column("No.", width=5, key="ordinal")
        self.add_column("Start", width=8, key="start")
        if show_speaker:
            self.add_column("Speaker", width=10, key="speaker")
        self.add_column("Source", width=30, key="source")
        self.add_column("Translation", width=30, key="translation")
        self._init_follow_tail()

    def on_resize(self, event: events.Resize) -> None:
        source_width, translation_width = subtitle_text_widths(
            self.scrollable_content_region.width,
            start_width=self.columns["start"].width,
            ordinal_width=self.columns["ordinal"].width,
            speaker_width=(
                self.columns["speaker"].width if self.show_speaker else 0
            ),
            cell_padding=self.cell_padding,
        )
        source = self.columns["source"]
        translation = self.columns["translation"]
        if source.width == source_width and translation.width == translation_width:
            return
        source.width = source_width
        translation.width = translation_width
        self._update_count += 1
        self._remeasure_rows(self.rows)

    def add_subtitle(self, cue: Cue) -> None:
        cells = [
            subtitle_cell(subtitle_ordinal(cue), no_wrap=True),
            subtitle_cell(format_start_time(cue.start), no_wrap=True),
        ]
        if self.show_speaker:
            cells.append(subtitle_cell(cue.speaker or ""))
        cells.append(subtitle_cell(cue.source))
        cells.append(subtitle_cell((cue.translated or "").strip()))
        self.add_row(*cells, height=None, key=cue.id)

    def update_subtitle(self, cue: Cue) -> None:
        self.update_cell(
            cue.id, "ordinal", subtitle_cell(subtitle_ordinal(cue), no_wrap=True)
        )
        self.update_cell(
            cue.id, "start", subtitle_cell(format_start_time(cue.start), no_wrap=True)
        )
        if self.show_speaker:
            self.update_cell(cue.id, "speaker", subtitle_cell(cue.speaker or ""))
        self.update_cell(cue.id, "source", subtitle_cell(cue.source))
        self.update_cell(
            cue.id,
            "translation",
            subtitle_cell((cue.translated or "").strip()),
        )
        self._remeasure_rows([cue.id])

    def _remeasure_rows(self, row_keys) -> None:
        # Textual auto-measures newly added rows, but currently exposes no
        # public API to re-measure an auto-height row after a cell update.
        for row_key in row_keys:
            row = self.rows.get(row_key)
            if row is None or not row.auto_height:
                continue
            row.height = 0
            self._new_rows.add(row.key)
        self._require_update_dimensions = True
        self.check_idle()
        self.refresh(layout=True)


class AgentConversationLog(FollowTailMixin, RichLog):
    """Agent trace log that follows new output only while the user wants it."""

    def __init__(self, **kwargs):
        super().__init__(auto_scroll=False, **kwargs)
        self._init_follow_tail()


class AgentInspector(ModalScreen[None]):
    """Live Agent list with a compact, backend-neutral conversation trace."""

    CSS = """
    AgentInspector { align: center middle; }
    #agent-dialog {
        width: 94%;
        height: 92%;
        background: $surface;
        border: round $accent;
        padding: 0 1;
    }
    #agent-inspector-title { height: 2; padding: 0 1; text-style: bold; }
    #agent-list { height: 13; }
    #agent-detail-title { height: 2; padding: 1 0 0 0; text-style: bold; }
    #agent-detail { height: 1fr; border: round $primary; padding: 0 1; }
    #agent-help { height: 1; color: $text-muted; }
    """
    BINDINGS = [
        Binding("escape", "dismiss", "Close", priority=True),
        Binding("f2", "dismiss", "Close", priority=True),
        Binding("up", "agent_up", "", show=False, priority=True),
        Binding("down", "agent_down", "", show=False, priority=True),
        Binding("pageup", "conversation_page_up", "", show=False, priority=True),
        Binding("pagedown", "conversation_page_down", "", show=False, priority=True),
        Binding("ctrl+home", "conversation_home", "", show=False, priority=True),
        Binding("ctrl+end", "conversation_end", "", show=False, priority=True),
    ]

    def __init__(self, tasks: dict[str, AgentTaskState]):
        super().__init__()
        self.tasks = tasks
        self.selected_operation: str | None = None
        self._rendered_detail: tuple[str | None, int, str] | None = None
        self._table_order: list[str] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="agent-dialog"):
            yield Static("Agents", id="agent-inspector-title")
            yield DataTable(
                zebra_stripes=True, cursor_type="row", id="agent-list"
            )
            yield Static("Conversation & Agent output", id="agent-detail-title")
            yield AgentConversationLog(wrap=True, markup=False, id="agent-detail")
            yield Static(
                "↑/↓ Select · Enter focus details · Tab switch focus · Esc/F2 close",
                id="agent-help",
            )

    def on_mount(self) -> None:
        table = self.query_one("#agent-list", DataTable)
        table.add_column("ID", key="id")
        table.add_column("State", key="state")
        table.add_column("Task", key="task")
        table.add_column("Attempt", key="attempt")
        table.add_column("Elapsed", key="elapsed")
        table.focus()
        self.update_tasks(force=True)
        self.set_interval(1, self.update_tasks)

    def _ordered_tasks(self) -> list[AgentTaskState]:
        active = sorted(
            (task for task in self.tasks.values() if task.active),
            key=lambda task: task.started_at or "",
        )
        finished = sorted(
            (task for task in self.tasks.values() if not task.active),
            key=lambda task: task.finished_at or task.started_at or "",
            reverse=True,
        )
        return active + finished

    def update_tasks(self, *, force: bool = False) -> None:
        table = self.query_one("#agent-list", DataTable)
        ordered = self._ordered_tasks()
        known = set(self._table_order)
        self._table_order.extend(
            task.operation_id for task in ordered if task.operation_id not in known
        )
        for operation_id in self._table_order:
            task = self.tasks.get(operation_id)
            if task is None:
                continue
            row = (
                task.display_id,
                task.state.title(),
                task.task_label,
                task.attempt_label,
                task.elapsed_label,
            )
            if operation_id not in table.rows:
                table.add_row(*row, key=operation_id)
                continue
            # Rewriting an unchanged cell still invalidates the DataTable
            # render cache, and finished tasks never change again.
            for column, value in zip(
                ("id", "state", "task", "attempt", "elapsed"), row
            ):
                if table.get_cell(operation_id, column) != value:
                    table.update_cell(operation_id, column, value)
        if self.selected_operation is None and self._table_order:
            self.selected_operation = self._table_order[0]
            table.move_cursor(row=0, column=0, animate=False)
        self.query_one("#agent-inspector-title", Static).update(
            f"Agents · {sum(task.active for task in ordered)} active · {len(ordered)} total"
        )
        self._render_detail(force=force)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key.value is None:
            return
        operation_id = str(event.row_key.value)
        changed = operation_id != self.selected_operation
        self.selected_operation = operation_id
        self._render_detail(force=changed)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key.value is not None:
            self.selected_operation = str(event.row_key.value)
            self._render_detail(force=True)
        self.query_one("#agent-detail", AgentConversationLog).focus()

    def _render_detail(self, *, force: bool = False) -> None:
        task = self.tasks.get(self.selected_operation or "")
        signature = (
            task.operation_id if task else None,
            len(task.events) if task else 0,
            task.state if task else "",
        )
        if not force and signature == self._rendered_detail:
            return
        previous_operation = self._rendered_detail[0] if self._rendered_detail else None
        previous_count = self._rendered_detail[1] if self._rendered_detail else 0
        self._rendered_detail = signature
        title = self.query_one("#agent-detail-title", Static)
        log = self.query_one("#agent-detail", AgentConversationLog)
        if task is None:
            title.update("Conversation & Agent output")
            log.follow_tail = True
            log.clear()
            log.write("No Agent activity yet.")
            log.restore_scroll_after_update(0)
            return
        model = f" · {task.model}" if task.model else ""
        title.update(
            f"{task.display_id} · {task.task_label} · {task.state.title()}{model}"
        )

        same_operation = signature[0] == previous_operation
        incremental = same_operation and previous_count <= len(task.events)
        if incremental:
            new_events = task.events[previous_count:]
            for event in new_events:
                self._write_event(log, event)
            if log.follow_tail and new_events:
                log.call_after_refresh(
                    log.scroll_end, animate=False, force=True, immediate=True
                )
            return

        # Selecting another Agent is a new conversation view, so start at
        # its latest event. Subsequent events append without rebuilding the
        # log, preserving an explicitly detached scroll position.
        log.follow_tail = True
        log.clear()
        for event in task.events:
            self._write_event(log, event)
        log.restore_scroll_after_update(0)

    def action_agent_up(self) -> None:
        detail = self.query_one("#agent-detail", AgentConversationLog)
        if detail.has_focus:
            detail.user_scroll_up()
        else:
            self.query_one("#agent-list", DataTable).action_cursor_up()

    def action_agent_down(self) -> None:
        detail = self.query_one("#agent-detail", AgentConversationLog)
        if detail.has_focus:
            detail.user_scroll_down()
        else:
            self.query_one("#agent-list", DataTable).action_cursor_down()

    def action_conversation_page_up(self) -> None:
        self.query_one("#agent-detail", AgentConversationLog).user_scroll_page_up()

    def action_conversation_page_down(self) -> None:
        self.query_one("#agent-detail", AgentConversationLog).user_scroll_page_down()

    def action_conversation_home(self) -> None:
        self.query_one("#agent-detail", AgentConversationLog).user_scroll_home()

    def action_conversation_end(self) -> None:
        self.query_one("#agent-detail", AgentConversationLog).user_scroll_end()

    @staticmethod
    def _write_event(log: AgentConversationLog, event: AgentTraceEvent) -> None:
        if event.kind == "user_message":
            log.write(f"You\n  {event.message}\n")
        elif event.kind == "agent_message":
            log.write(f"Agent\n  {event.message}\n")
        elif event.kind == "agent_output":
            log.write(f"Agent output · {event.message}")
            if event.detail:
                log.write(event.detail)
            log.write("")
        elif event.kind == "thinking":
            log.write("Agent · Thinking…")
        elif event.kind == "tool_start":
            log.write(f"● {event.message}")
            if event.detail:
                log.write(f"  {event.detail}")
        elif event.kind == "tool_result":
            log.write(f"└ {event.message}")
            if event.detail:
                log.write(f"  {event.detail}\n")
        elif event.kind == "result":
            log.write(f"Agent\n  {event.message}")
            if event.detail:
                log.write(f"  {event.detail}")
            log.write("")
        elif event.kind == "error":
            log.write(f"Error · {event.message}")
        elif event.kind == "lifecycle" and event.state == "retrying":
            log.write(f"Status · {event.message}")
        elif event.kind == "lifecycle" and event.state in {"failed", "cancelled"}:
            log.write(f"Status · {event.state.title()} · {event.message}")


class AlignmentModelFailureDialog(ModalScreen[AlignmentModelFailureDecision]):
    """Require an explicit choice after a WhisperX dependency download failure."""

    CSS = """
    AlignmentModelFailureDialog { align: center middle; }
    #alignment-failure-dialog {
        width: 76;
        max-width: 94%;
        height: auto;
        background: $surface;
        border: round $error;
        padding: 1 2;
    }
    #alignment-failure-title { height: 2; text-style: bold; color: $error; }
    #alignment-failure-message { height: auto; max-height: 12; margin-bottom: 1; }
    #alignment-failure-help { height: auto; color: $text-muted; margin-bottom: 1; }
    #alignment-failure-actions { height: 3; align-horizontal: right; }
    #alignment-failure-actions Button { margin-left: 1; }
    """

    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="alignment-failure-dialog"):
            yield Static("WhisperX alignment setup failed", id="alignment-failure-title")
            yield Static(Text(self.message), id="alignment-failure-message")
            yield Static(
                "Retry attempts the dependency download/load again. Fallback "
                "explicitly continues this job with VAD alignment.",
                id="alignment-failure-help",
            )
            with Horizontal(id="alignment-failure-actions"):
                yield Button("Fall back to VAD", id="alignment-fallback")
                yield Button("Retry", variant="primary", id="alignment-retry")

    def on_mount(self) -> None:
        self.query_one("#alignment-retry", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        decision: AlignmentModelFailureDecision = (
            "retry" if event.button.id == "alignment-retry" else "fallback"
        )
        self.dismiss(decision)


class StreamCompletionDialog(ModalScreen[str]):
    def __init__(self, available: bool):
        super().__init__()
        self.available = available

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Capture complete. Choose how to finish the recording.")
            yield Button("直接进入 review", id="review", disabled=not self.available)
            yield Button("使用非 stream 重新转录", id="retranscribe", variant="primary")
            yield Button("退出并保留工作目录", id="exit")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id)


class YakiFlowApp(App[None]):
    """Progress, recent subtitles, and post-publication review in one TUI."""

    CSS = """
    #agent-status { height: 1; margin: 0 1; color: $text-muted; }
    #status { height: 3; padding: 1; }
    #progress-label { height: 1; margin: 0 1; color: $text-muted; }
    #progress { margin: 0 1; }
    #recent {
        height: 1fr;
        border: round $accent;
        padding: 0 1;
        overflow-x: hidden;
        overflow-y: scroll;
    }
    #conversation { height: 12; border: round $primary; }
    """
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("s", "stop", "Stop", priority=True),
        Binding("f2", "agents", "Agents", priority=True),
        Binding("o", "open_media", "Open media", priority=True),
        Binding("up", "subtitles_up", "", show=False, priority=True),
        Binding("down", "subtitles_down", "", show=False, priority=True),
        Binding("pageup", "subtitles_page_up", "", show=False, priority=True),
        Binding("pagedown", "subtitles_page_down", "", show=False, priority=True),
        Binding("ctrl+home", "subtitles_home", "", show=False, priority=True),
        Binding("ctrl+end", "subtitles_end", "", show=False, priority=True),
    ]

    def __init__(self, job: YakiFlowJob):
        super().__init__()
        self.job = job
        self.succeeded = False
        self.whisper_cues = 0
        self.processed_ids: set[str] = set()
        self.agent_tasks: dict[str, AgentTaskState] = {}
        self._agent_display_sequence = 0
        self._drive_worker: Worker[None] | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static("Agent · Idle", id="agent-status")
            yield Static("Current stage · Starting…", id="status")
            yield Static("Overall progress", id="progress-label")
            yield PipelineProgressBar(total=100, id="progress")
            recent = SubtitleTable(
                id="recent", show_speaker=self.job.diarization_enabled
            )
            recent.border_title = "Subtitles"
            recent.border_subtitle = "↑/↓ · PgUp/PgDn · ^Home/^End"
            yield recent
            conversation = ReadOnlyRichLog(id="conversation", wrap=True, markup=False)
            conversation.border_title = "Pipeline messages"
            yield conversation
        yield Footer()

    def on_mount(self) -> None:
        self.job.listener = self._event
        self.job.stream_completion_listener = self._choose_stream_completion
        self.job.alignment_model_failure_listener = (
            self._choose_alignment_model_failure
        )
        # The subtitle timeline is the primary keyboard-scroll target. A
        # focusable pipeline log appears later in the layout and Textual
        # would otherwise focus it automatically, sending the global
        # arrow/page bindings to the log instead of this panel.
        self.query_one("#recent", SubtitleTable).focus()
        self._load_existing_subtitles()
        self._drive_worker = self.run_worker(self._drive(), exclusive=True)

    def _load_existing_subtitles(self) -> None:
        """Load the durable subtitle timeline into the subtitle table."""
        table = self.query_one("#recent", SubtitleTable)
        # The database orders by insertion ordinal, which mid-draft on a
        # resumed word-mode job is batch-completion order, not time order.
        for cue in sorted(
            self.job.db.list_cues(stable_only=True),
            key=lambda cue: (cue.start, cue.end, cue.speaker or ""),
        ):
            table.add_subtitle(cue)
            self.whisper_cues += 1
            if cue.translated:
                self.processed_ids.add(cue.id)

    def on_unmount(self) -> None:
        self.job.listener = None
        self.job.alignment_model_failure_listener = None

    async def _choose_stream_completion(self, available: bool) -> str:
        return await self.push_screen_wait(StreamCompletionDialog(available))

    async def _choose_alignment_model_failure(
        self, message: str
    ) -> AlignmentModelFailureDecision:
        return await self.push_screen_wait(AlignmentModelFailureDialog(message))

    def _save_traceback(self, details: str) -> str:
        path = self.job.work_dir / "yakiflow-traceback.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(details.rstrip() + "\n\n")
        return str(path)

    async def _drive(self) -> None:
        try:
            paths = await self.job.run()
            self.succeeded = True
            log = self.query_one("#conversation", RichLog)
            log.write("Published:\n" + "\n".join(str(path) for path in paths))
            log.write(
                "Continuing in the interactive Agent for review, refinement, and memory…"
            )
            # Leave the progress TUI before handing the terminal to Codex
            # or Claude Code. The CLI performs that hand-off after run()
            # returns, so the two interfaces never compete for the TTY.
            self.exit()
        except StreamExit:
            self.exit()
        except asyncio.CancelledError:
            if self.is_running:
                self.query_one("#conversation", RichLog).write(
                    f"Interrupted; resume {self.job.work_dir}"
                )
        except Exception as exc:
            # Keep the concise failure message, but include the complete
            # exception chain so timing-pass errors are diagnosable from
            # the TUI itself.
            log = self.query_one("#conversation", RichLog)
            details = traceback.format_exc().rstrip()
            log.write(f"Failed: {exc}", scroll_end=True)
            path = self._save_traceback(details)
            log.write(f"Full traceback saved to: {path}", scroll_end=True)
            log.scroll_end(immediate=True, x_axis=False)

    async def _event(self, event: JobEvent) -> None:
        # Agent activity contexts emit their final Idle event while a
        # canceled worker unwinds. The screen may already be unmounted.
        if not self.is_running:
            return
        if event.agent_trace is not None:
            self._record_agent_trace(event.agent_trace)
            return
        if event.kind == "agent":
            self.query_one("#agent-status", Static).update(event.message)
            return
        if event.progress is not None:
            progress = self.query_one("#progress", PipelineProgressBar)
            progress.update(progress=event.progress * 100)
            progress.set_estimated_remaining(event.estimated_remaining)
        if event.kind == "timeline-replaced":
            table = self.query_one("#recent", SubtitleTable)
            previous_y = table.scroll_y
            table.clear(columns=False)
            self.whisper_cues = 0
            self.processed_ids.clear()
            if event.cues is not None:
                for cue in event.cues:
                    table.add_subtitle(cue)
                    self.whisper_cues += 1
                    if cue.translated:
                        self.processed_ids.add(cue.id)
                status = (
                    f"Agent processed: {len(self.processed_ids)}"
                    f" / {self.whisper_cues} subtitles"
                )
            else:
                self._load_existing_subtitles()
                status = f"aligned timeline: {self.whisper_cues} subtitles"
            table.restore_scroll_after_update(previous_y)
            self.query_one("#status", Static).update(f"Current stage · {status}")
            return
        if event.kind in {"transcript", "subtitle"} and event.cue:
            table = self.query_one("#recent", SubtitleTable)
            is_new = event.cue.id not in table.rows
            if is_new:
                self.whisper_cues += 1
            previous_y = table.scroll_y
            if is_new:
                table.add_subtitle(event.cue)
            else:
                table.update_subtitle(event.cue)
            if event.kind == "subtitle":
                self.processed_ids.add(event.cue.id)
                status = f"Agent processed: {len(self.processed_ids)} / Whisper: {self.whisper_cues}"
            else:
                status = f"Whisper subtitles: {self.whisper_cues} · waiting for Agent"
            self.query_one("#status", Static).update(f"Current stage · {status}")
            table.restore_scroll_after_update(previous_y)
        else:
            self.query_one("#status", Static).update(f"Current stage · {event.message}")
        if event.kind in {"warning", "failed"}:
            log = self.query_one("#conversation", RichLog)
            log.write(f"[{event.kind}] {event.message}")
            if event.error_traceback:
                path = self._save_traceback(event.error_traceback)
                log.write(f"Full traceback saved to: {path}", scroll_end=True)
                log.scroll_end(immediate=True, x_axis=False)

    def _record_agent_trace(self, event: AgentTraceEvent) -> None:
        task = self.agent_tasks.get(event.operation_id)
        if task is None:
            self._agent_display_sequence += 1
            task = AgentTaskState(
                event.operation_id,
                f"A{self._agent_display_sequence:02d}",
            )
            self.agent_tasks[event.operation_id] = task
        task.apply(event)
        if isinstance(self.screen, AgentInspector):
            self.screen.update_tasks(force=True)

    def action_agents(self) -> None:
        if isinstance(self.screen, AgentInspector):
            self.screen.dismiss()
            return
        self.push_screen(AgentInspector(self.agent_tasks))

    def action_quit(self) -> None:
        self.exit()

    def action_stop(self) -> None:
        worker = self._drive_worker
        if worker is None or worker.is_finished:
            self.notify("No running job to stop.")
            return
        self.query_one("#status", Static).update("Current stage · Stopping…")
        self.query_one("#conversation", RichLog).write(
            "Stop requested; preserving completed work and finalizing available media…"
        )
        worker.cancel()

    def action_open_media(self) -> None:
        if open_media(
            self.job.settings.review.video_open_command,
            self.job.media_path,
            self.job.subtitle_path,
            cwd=self.job.work_dir,
        ):
            self.notify("Opened media with subtitles.")
        else:
            self.notify("Media or subtitles are not ready (or mpv is unavailable).")

    def action_subtitles_up(self) -> None:
        if isinstance(self.screen, AgentInspector):
            self.screen.action_agent_up()
            return
        conversation = self.query_one("#conversation", ReadOnlyRichLog)
        if conversation.has_focus:
            conversation.scroll_up()
            return
        self.query_one("#recent", SubtitleTable).user_scroll_up()

    def action_subtitles_down(self) -> None:
        if isinstance(self.screen, AgentInspector):
            self.screen.action_agent_down()
            return
        conversation = self.query_one("#conversation", ReadOnlyRichLog)
        if conversation.has_focus:
            conversation.scroll_down()
            return
        self.query_one("#recent", SubtitleTable).user_scroll_down()

    def action_subtitles_page_up(self) -> None:
        if isinstance(self.screen, AgentInspector):
            self.screen.action_conversation_page_up()
            return
        conversation = self.query_one("#conversation", ReadOnlyRichLog)
        if conversation.has_focus:
            conversation.scroll_page_up()
            return
        self.query_one("#recent", SubtitleTable).user_scroll_page_up()

    def action_subtitles_page_down(self) -> None:
        if isinstance(self.screen, AgentInspector):
            self.screen.action_conversation_page_down()
            return
        conversation = self.query_one("#conversation", ReadOnlyRichLog)
        if conversation.has_focus:
            conversation.scroll_page_down()
            return
        self.query_one("#recent", SubtitleTable).user_scroll_page_down()

    def action_subtitles_home(self) -> None:
        if isinstance(self.screen, AgentInspector):
            self.screen.action_conversation_home()
            return
        conversation = self.query_one("#conversation", ReadOnlyRichLog)
        if conversation.has_focus:
            conversation.scroll_home()
            return
        self.query_one("#recent", SubtitleTable).user_scroll_home()

    def action_subtitles_end(self) -> None:
        if isinstance(self.screen, AgentInspector):
            self.screen.action_conversation_end()
            return
        conversation = self.query_one("#conversation", ReadOnlyRichLog)
        if conversation.has_focus:
            conversation.scroll_end(immediate=True)
            return
        self.query_one("#recent", SubtitleTable).user_scroll_end()


def run(job: YakiFlowJob) -> bool:
    app = YakiFlowApp(job)
    app.run()
    return app.succeeded
