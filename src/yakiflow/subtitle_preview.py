from __future__ import annotations

import sys
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding

from .media_player import open_media
from .models import Cue
from .subtitles import parse_ass
from .tui import SubtitleTable


def read_subtitle(path: Path) -> list[Cue]:
    """Read a subtitle file for preview, skipping any event that cannot parse."""
    if not path.is_file():
        return []
    events, _problems = parse_ass(
        path.read_text(encoding="utf-8", errors="replace")
    )
    cues: list[Cue] = []
    # Number by position: an in-progress Agent edit has no stable IDs to key
    # on, and the subtitle table only needs unique row keys. Timings are
    # clamped for the same reason: an event that ends before it starts is
    # reported at publish time, not crashed on here.
    for position, event in enumerate(events, 1):
        first, *rest = event.text_lines
        translated = first if rest else None
        source = "\n".join(rest) if rest else first
        cues.append(
            Cue(
                str(position),
                event.start,
                max(event.start, event.end),
                source,
                translated,
                speaker=event.name or None,
            )
        )
    return cues


class SubtitlePreviewApp(App[None]):
    CSS = """
    Screen { padding: 0; }
    #recent { width: 1fr; height: 1fr; }
    """

    BINDINGS = [
        Binding("o", "open_media", "Open media", priority=True),
    ]

    def __init__(
        self,
        path: Path,
        marker: Path,
        media_path: Path | None = None,
        video_open_command: str | None = None,
        show_speaker: bool = False,
    ):
        super().__init__()
        self.path = path
        self.marker = marker
        self.media_path = media_path
        self.video_open_command = video_open_command
        self.show_speaker = show_speaker
        self._signature: tuple[int, int] | None = None

    def compose(self) -> ComposeResult:
        table = SubtitleTable(id="recent", show_speaker=self.show_speaker)
        table.border_title = "Live subtitles"
        table.border_subtitle = "↑/↓ · PgUp/PgDn · ^Home/^End"
        yield table

    def action_open_media(self) -> None:
        if open_media(
            self.video_open_command,
            self.media_path,
            self.path,
            cwd=self.path.parent,
        ):
            self.notify("Opened media with subtitles.")
        else:
            self.notify("Media or subtitles are not ready (or mpv is unavailable).")

    def on_mount(self) -> None:
        self.set_interval(0.4, self._refresh)
        self._refresh()

    def _refresh(self) -> None:
        if self.marker.exists():
            self.exit()
            return
        try:
            stat = self.path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            signature = None
        if signature == self._signature:
            return
        self._signature = signature
        table = self.query_one("#recent", SubtitleTable)
        table.replace_subtitles(read_subtitle(self.path))


def main() -> int:
    if len(sys.argv) != 6:
        return 2
    media = Path(sys.argv[3]) if sys.argv[3] else None
    command = sys.argv[4] or None
    SubtitlePreviewApp(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        media,
        command,
        show_speaker=bool(sys.argv[5]),
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
