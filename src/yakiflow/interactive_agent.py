"""The terminal hand-off used for all user-directed subtitle work.

YakiFlow's batch agents return structured data and are intentionally not
interactive.  Review, final polishing, and memory maintenance are different:
they need a conversation with the user and may require several iterations.
This module only starts the configured Agent CLI in the job directory; the
Agent owns that conversation and the resulting file edits.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import sys
import uuid
from pathlib import Path
from typing import Sequence

from .config import Settings
from .database import JobDatabase
from .media_player import open_media
from .models import OutputMode
from .process import CommandRunner, ProcessError, ProcessResult


class AgentFileDisplay:
    """A best-effort live, read-only view of the Agent's staging subtitle."""

    def __init__(self, marker: Path | None = None):
        self.marker = marker

    async def close(self) -> None:
        if self.marker is not None:
            self.marker.touch()


def _preview_command(
    file_path: Path,
    marker: Path,
    media_path: Path | None = None,
    video_open_command: str | None = None,
    show_speaker: bool = False,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "yakiflow.subtitle_preview",
        str(file_path),
        str(marker),
        str(media_path) if media_path else "",
        video_open_command or "",
        "1" if show_speaker else "",
    ]


def _source_media(work_dir: Path) -> Path | None:
    db = JobDatabase(work_dir / "job.sqlite3")
    try:
        return db.artifact("source_media")
    finally:
        db.close()


def review_open_argv(command: str, file_path: Path, work_dir: Path) -> list[str]:
    """Expand a review-display command without invoking a shell."""
    has_srt = any(
        placeholder in command
        for placeholder in ("{subtitle}", "{srt}", "{file}", "{srt_file}")
    )
    memory_file = work_dir / "memory.md"
    replacements = {
        "{subtitle}": file_path,
        # The srt spellings predate the ASS output and keep working.
        "{srt}": file_path,
        "{file}": file_path,
        "{srt_file}": file_path,
        "{workdir}": work_dir,
        "{memory}": memory_file,
        "{memory_file}": memory_file,
    }
    for placeholder, path in replacements.items():
        command = command.replace(placeholder, shlex.quote(str(path)))
    argv = shlex.split(command)
    if not has_srt:
        argv.append(str(file_path))
    return argv


def _split_window_args() -> list[str]:
    """Return the tmux split-window argv for the current terminal shape."""
    wide = shutil.get_terminal_size(fallback=(100, 24)).columns >= 120
    args = ["tmux", "split-window", "-h" if wide else "-v"]
    if not wide:
        args.append("-b")
    return args


async def start_agent_file_display(
    settings: Settings,
    work_dir: Path,
    outputs: Sequence[Path],
) -> AgentFileDisplay:
    """Open a live SRT preview using the configured display mode."""
    if not outputs:
        return AgentFileDisplay()
    file_path = outputs[0]
    media_path = _source_media(work_dir)
    marker = work_dir / ".agent-display-done"
    marker.unlink(missing_ok=True)
    if settings.review.display_mode in {"open", "both"}:
        command_text = settings.review.open_command
        if not command_text:
            raise ValueError(
                "review_open_command is required for open or both display mode"
            )
        # An optional preview must never turn into a failed review: a missing
        # viewer here would otherwise abort before the interactive Agent runs
        # and before the staged subtitles are finalized. Report it, though — a
        # silently unusable review_open_command leaves nothing to diagnose.
        try:
            command = review_open_argv(command_text, file_path, work_dir)
            await asyncio.create_subprocess_exec(
                *command,
                cwd=work_dir,
                # The viewer shares the Agent's terminal otherwise, where its
                # own output overwrites the rendering, and the user's Ctrl-C
                # would reach it instead of only the Agent.
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
        except (OSError, IndexError, ValueError) as exc:
            print(f"yakiflow: review_open_command failed: {exc}", file=sys.stderr)
        if settings.review.display_mode == "open":
            return AgentFileDisplay()

    # A split pane only makes sense from inside an active multiplexer
    # session. If tmux has no server/socket, leave the Agent terminal alone
    # instead of turning an optional preview into a failed review.
    if not os.environ.get("TMUX"):
        return AgentFileDisplay()
    command = _split_window_args()
    command.extend(("--", *_preview_command(
        file_path, marker, media_path, settings.review.video_open_command,
        settings.diarization_enabled,
    )))
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=work_dir,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError:
        return AgentFileDisplay()
    await process.wait()
    if process.returncode:
        return AgentFileDisplay()
    return AgentFileDisplay(marker)


def build_interactive_prompt(
    settings: Settings,
    work_dir: Path,
    outputs: Sequence[Path] = (),
    context_files: Sequence[Path] = (),
    previous_problems: Sequence[str] = (),
) -> str:
    output_list = "\n".join(f"- {path}" for path in outputs) or (
        "- inspect the published subtitle files in the work directory"
    )
    memory = work_dir / "memory.md"
    context_section = ""
    if context_files:
        context_list = "\n".join(f"- {path}" for path in context_files)
        context_section = f"""
Additional reference files were copied into the work directory for this review:
{context_list}

Read these files when they are useful as supporting context. They are reference
material only: do not edit or publish them.
"""
    previous_problems_section = ""
    if previous_problems:
        problem_list = "\n".join(f"- {problem}" for problem in previous_problems)
        previous_problems_section = f"""
The previous review session left these publish-blocking defects behind, and
YakiFlow will re-check them when it publishes. Fix them first:
{problem_list}
"""
    output_mode = OutputMode(settings.output_mode)
    conversation_language = settings.target_language or settings.source_language
    language_rule = (
        f"""Write to the user in {conversation_language}, including the opening review
report. If the user writes to you in another language, switch to that language
and keep using it for the rest of the session."""
        if conversation_language
        else """Write to the user in whatever language they use, and keep using
it for the rest of the session."""
    )
    review_scope = {
        OutputMode.SOURCE: f"""The final deliverable is a source-language monolingual subtitle ({settings.source_language or 'the detected source language'}).
Fix clear ASR/transcription errors, source-text omissions or duplication,
punctuation, segmentation, formatting, timing, and source-term consistency.
Do not add a translation or a second language line to the final subtitle.""",
        OutputMode.TRANSLATED: f"""The final deliverable is a target-language monolingual subtitle ({settings.target_language or 'the configured target language'}).
Fix mistranslations, missing or duplicated meaning, unnatural phrasing,
terminology/style inconsistency, target-language punctuation and formatting,
and timing. Use source or transcription data as evidence when needed, but keep
the final subtitle target-language-only; do not add source-language lines.""",
        OutputMode.BILINGUAL: f"""The final deliverable is a bilingual subtitle: target language ({settings.target_language or 'the configured target language'}) first and source language ({settings.source_language or 'the detected source language'}) second in each cue.
Fix both clear source transcription problems and target translation problems,
including omissions, duplication, unnatural target phrasing, terminology,
punctuation, formatting, and timing. Preserve that target-first/source-second
layout and make each pair say the same thing.""",
        OutputMode.ALL: f"""The final deliverables include source-language monolingual, target-language monolingual, and bilingual subtitles ({settings.source_language or 'the detected source language'} -> {settings.target_language or 'the configured target language'}).
Fix both clear source transcription problems and target translation problems,
including omissions, duplication, unnatural target phrasing, terminology,
punctuation, formatting, and timing. Mirror every applicable text and timing
correction across all three artifacts so they remain equivalent; keep the
bilingual file target-first and source-second.""",
    }[output_mode]
    merge_mirroring = (
        "\nApply every merge to the source side and to all other configured"
        "\nartifacts as well, so the outputs stay cue-for-cue equivalent."
        if output_mode in {OutputMode.BILINGUAL, OutputMode.ALL}
        else ""
    )
    artifact_invariant = (
        "\nAll three artifacts must keep the same event count and identical"
        "\nevent timings and Name fields as each other."
        if output_mode is OutputMode.ALL
        else ""
    )
    bilingual_layout = (
        """
Bilingual layout: each Dialogue event carries both languages in one Text
field, translation first and source second, separated by `\\N`."""
        if output_mode in {OutputMode.BILINGUAL, OutputMode.ALL}
        else ""
    )
    phrasing_scope = "" if output_mode is OutputMode.SOURCE else f"""
Phrasing quality for the target language ({settings.target_language or 'the configured target'}):
Every target line must read as if it had been written in that language, not
translated into it. Rewrite literal word-by-word renderings, source word order
carried over unchanged, calqued idioms, and other translationese into the
wording a native speaker would actually use, and keep the register natural for
spoken subtitles. Do not change the meaning, speaker intent, or tone to achieve
this, and do not add or drop content to make a line read better. Terminology and
style rules in {memory} still win over your own phrasing preference. If the
source makes sense, fix misleading or awkward translation; do not smooth over
a possible source recognition error by making the translation sound natural.

Merging cues the target language cannot keep apart:
One sentence is often split across several cues, and the target language may
reorder or regroup it so that no single translated cue can carry a complete,
self-contained meaning on its own. Merge those cues into one when all of the
following hold: they belong to the same sentence and the same speaker, the
merged text is still short enough to read comfortably within its on-screen
time, and they are not separated by a long pause. To merge in ASS, extend one
Dialogue event's times to cover the whole span and delete the other Dialogue
line. Merging is the exception, not the default: when each cue can stand on
its own after a natural rewrite, keep the original split and timing.{merge_mirroring}
"""
    return f"""You are the interactive YakiFlow subtitle editor.

Work directly in {work_dir}. The batch pipeline has finished and published these
subtitle artifacts:
{output_list}

The staged subtitle files in this work directory are the files to inspect and
edit. YakiFlow will publish your edits after this session and remove the
destination-side draft snapshot.
{context_section}{previous_problems_section}

The source language is {settings.source_language or 'detected automatically'} and
the target language is {settings.target_language or 'the configured target'}.
{language_rule}

Review and editing scope for the configured final output:
{review_scope}
{phrasing_scope}
Subtitle file format:
The staged files are ASS (Advanced SubStation Alpha) documents. Your editing
surface is the `Dialogue:` lines in the `[Events]` section; the header sections
above it are regenerated wholesale at publish time, so changing them has no
effect on what is published. Each Dialogue's Name field is the diarized
speaker label: leave it unchanged unless the user asks for a speaker change.
Line breaks inside a subtitle are written as `\\N` in the Text field.{bilingual_layout}
Cues of two different named speakers may legitimately overlap in time — that
is simultaneous speech, not an error — so never merge, delete, or retime cues
just to remove such an overlap.{artifact_invariant}

Start immediately with an autonomous review before waiting for user input:
inspect every staged subtitle and {memory}, follow the output-specific scope
above, and check the complete timeline for relevant text, timing, formatting,
and cross-cue consistency problems. Actively flag semantically unnatural or
abruptly out-of-place wording, even when grammatically valid. Read neighboring
cues together and use the topic, speaker intent, reference material, and memory
to distinguish recognition errors from translation problems or intentional
unusual wording. For suspected recognition errors, consider homophones and
similarly pronounced words; propose source corrections only when both context
and a close phonetic match support them, not contextual plausibility alone.
Report the initial findings in the terminal
and apply only clearly safe mechanical fixes before asking the user questions.
If a cue or timing is genuinely ambiguous, you may occasionally consult the raw
transcription artifacts in the work directory (for example transcriber JSON
output or the `process_logs` table in `job.sqlite3`), but do not routinely
reread raw output when the published subtitles already provide enough evidence.

Then work with the user conversationally until they are satisfied. Handle all
of the following in this one session:

1. Review: ask for and understand natural-language subtitle feedback. Inspect
   the staged subtitles and any source/translation evidence applicable to the
   output-specific scope yourself; do not require the user to provide cue IDs
   or line numbers. Whenever you point out or discuss a specific subtitle
   sentence/cue in the terminal, include its start and end timestamp
   (for example, `0:01:23.45 - 0:01:25.00`). If wording or meaning remains
   uncertain, keep the text unchanged and ask a focused question with the
   current wording, plausible alternatives, and reasons for each. Explain
   whether the uncertainty concerns recognition or translation; do not guess.
2. Refinement: apply agreed corrections to the subtitle artifacts, preserving
   the ASS structure and any timing or formatting that does not need
   correction, and check the complete timeline for consistency. When correcting
   source text, update its translation too, within the configured output scope.
3. Memory: proactively extract knowledge that will remain useful for future
   subtitle jobs and is not already in {memory}. That file may not exist yet:
   treat a missing one as empty memory and create it when you have an approved
   item to write. Do this from your own review of the subtitles and corrections
   as well as from user comments; do not wait for a comment to mention a memory
   item. Focus on:
   - canonical names, proper nouns, domain terms, acronyms, and their preferred
     target renderings, capitalization, or explicit do-not-translate rules;
   - recurring ASR corrections or domain meanings that can resolve the same
     homophone, spelling, or ambiguity in later material;
   - durable translation and style preferences such as tone/register,
     honorifics and forms of address, locale/orthography, punctuation, and
     number, date, or unit conventions;
   - stable project or series context that affects repeated translation choices,
     such as speaker identities, relationships, or naming conventions.
   Exclude duplicates of existing memory, one-off cue fixes, plot details with
   no likely future use, obvious general language facts, and uncertain guesses.
   Present each new candidate and its reuse rationale to the user, obtain
   explicit approval, and only then write the approved item to {memory}. If
   nothing is genuinely reusable, say so rather than inventing candidates. Do
   not merely describe approved subtitle or memory edits--make them in the files.
   Whenever the user asks to add, remove, or change anything in {memory},
   re-check every staged subtitle artifact against the resulting memory for
   inconsistent terminology, style, names, or other applicable guidance. Apply
   all necessary subtitle corrections in the same session (mirroring them
   across the configured output artifacts); do not stop after editing memory.

Exactly two kinds of file are yours to edit: the staged subtitle artifacts
listed above and {memory}. Everything else in the work directory is read-only
evidence, including `job.sqlite3` and the transcriber artifacts: editing them
changes nothing about what YakiFlow publishes and can corrupt the job's resume
state. Keep the staged subtitle outputs and memory internally consistent.
When the user says the session is complete, summarize the changes and exit.
"""


def build_interactive_command(settings: Settings, prompt: str) -> list[str]:
    """Build the configured Agent CLI command in interactive mode."""
    final = settings.agent.final
    model = final.model or ""
    effort = final.effort or "high"
    extra_options = final.extra_options or ()
    if final.backend == "codex":
        command = [
            "codex",
            "--sandbox",
            "workspace-write",
        ]
        if model:
            command.extend(("--model", model))
        command.extend(("--config", f'model_reasoning_effort="{effort}"'))
        command.extend(extra_options)
        command.append(prompt)
        return command
    if final.backend == "claude":
        command = ["claude"]
        if model:
            command.extend(("--model", model))
        command.extend(("--effort", effort))
        command.extend(extra_options)
        command.append(prompt)
        return command
    raise ValueError(f"unsupported agent backend: {final.backend}")


def build_memory_conflict_prompt(
    work_dir: Path,
    destination: Path,
    diff: str,
) -> str:
    memory = work_dir / "memory.md"
    return f"""You are resolving a concurrent YakiFlow translation-memory update.

Work directly in {work_dir}. The staged memory file is {memory}. While this job
was running, the durable destination at {destination} changed. Integrate the
destination-side changes shown below into the staged memory without losing the
valid edits already made in the staged file.

Inspect and edit only the staged memory file; do not edit the destination. Apply
the resolution in the file, preserve valid Markdown structure, avoid duplicate
memory entries, briefly summarize the merge, and exit. If the intent of a
change is genuinely ambiguous, ask the user before choosing.

<destination-memory-diff>
{diff.rstrip()}
</destination-memory-diff>
"""


async def run_memory_conflict_agent(
    settings: Settings,
    work_dir: Path,
    destination: Path,
    diff: str,
    *,
    runner: CommandRunner | None = None,
) -> ProcessResult:
    """Launch an interactive Agent to merge one destination memory revision."""
    runner = runner or CommandRunner()
    prompt = build_memory_conflict_prompt(work_dir, destination, diff)
    command = build_interactive_command(settings, prompt)
    return await runner.run_interactive(command, cwd=work_dir)


async def run_interactive_agent(
    settings: Settings,
    work_dir: Path,
    outputs: Sequence[Path] = (),
    *,
    context_files: Sequence[Path] = (),
    previous_problems: Sequence[str] = (),
    runner: CommandRunner | None = None,
) -> ProcessResult:
    """Hand the terminal to the configured interactive Agent CLI."""
    runner = runner or CommandRunner()
    prompt = build_interactive_prompt(
        settings, work_dir, outputs, context_files, previous_problems
    )
    command = build_interactive_command(settings, prompt)
    if settings.review.auto_open_video and outputs:
        open_media(
            settings.review.video_open_command,
            _source_media(work_dir),
            outputs[0],
            cwd=work_dir,
        )
    if (
        settings.review.display_mode in {"split", "both"}
        and not os.environ.get("TMUX")
    ):
        return await _run_in_new_tmux_session(
            command, work_dir, outputs, _source_media(work_dir),
            settings.review.video_open_command, settings.diarization_enabled,
        )
    return await runner.run_interactive(command, cwd=work_dir)


async def _run_in_new_tmux_session(
    command: Sequence[str],
    work_dir: Path,
    outputs: Sequence[Path],
    media_path: Path | None = None,
    video_open_command: str | None = None,
    show_speaker: bool = False,
) -> ProcessResult:
    """Run Agent and preview in a tmux session created by YakiFlow itself."""
    session = f"yakiflow-review-{uuid.uuid4().hex[:8]}"
    marker = work_dir / ".agent-display-done"
    status_file = work_dir / ".agent-display-status"
    marker.unlink(missing_ok=True)
    status_file.unlink(missing_ok=True)
    file_path = outputs[0] if outputs else work_dir / "memory.md"
    preview = _preview_command(
        file_path, marker, media_path, video_open_command, show_speaker
    )
    wrapped_agent = (
        f"{shlex.join(command)}; status=$?; "
        f"printf '%s' \"$status\" > {shlex.quote(str(status_file))}; "
        f"touch {shlex.quote(str(marker))}; "
        f"tmux kill-session -t {shlex.quote(session)}; exit \"$status\""
    )
    try:
        created = await asyncio.create_subprocess_exec(
            "tmux", "new-session", "-d", "-s", session, "-c", str(work_dir),
            "--", "sh", "-lc", wrapped_agent,
            cwd=work_dir,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await created.wait()
        if created.returncode:
            return await CommandRunner().run_interactive(command, cwd=work_dir)
        split_args = _split_window_args()
        split = await asyncio.create_subprocess_exec(
            *split_args, "-t", session, "-c", str(work_dir),
            "--", *preview,
            cwd=work_dir,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await split.wait()
        attached = await asyncio.create_subprocess_exec(
            "tmux", "attach-session", "-t", session, cwd=work_dir
        )
        await attached.wait()
    except OSError:
        return await CommandRunner().run_interactive(command, cwd=work_dir)
    try:
        returncode = int(status_file.read_text(encoding="utf-8") or "1")
    except (OSError, ValueError):
        returncode = attached.returncode or 1
    result = ProcessResult(tuple(command), returncode, "", "")
    if returncode:
        raise ProcessError(result)
    return result
