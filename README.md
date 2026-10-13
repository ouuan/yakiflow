# YakiFlow

[简体中文](README.zh-CN.md) | English

YakiFlow turns a local video, audio file, web video, or live stream into an ASS
subtitle file. It transcribes with `whisper.cpp` or the ElevenLabs Scribe API,
translates with Codex or Claude Code, and lets you review the result with an
interactive Agent before it is saved. With a diarizing backend, each subtitle
carries its speaker, and two speakers talking at once become two overlapping
subtitles.

Jobs are resumable: if a run is interrupted, YakiFlow keeps its completed work
and prints the command needed to continue.

## Quick start

### 1. Install the required tools

You need:

- Python 3.12 or newer;
- `ffmpeg`, together with the `ffprobe` that ships with it;
- either the `codex` or `claude` CLI, installed and signed in;
- for the default local transcription, a recent
  [whisper.cpp](https://github.com/ggml-org/whisper.cpp) build whose
  `whisper-cli` executable is on `PATH`.

Install `yt-dlp` as well if you want to process URLs. `tmux` and `mpv` are
optional conveniences for subtitle review and video preview. The ElevenLabs
backends need no local transcription tools at all — see
[Transcription backends](#transcription-backends).

### 2. Install YakiFlow

From this repository:

```console
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Add `.[elevenlabs]` to use the ElevenLabs backends and `.[keyring]` to store
the API key in the OS keyring:

```console
python -m pip install -e '.[elevenlabs,keyring]'
```

### 3. Save your usual settings

Create `yakiflow.toml` in the directory where you will run YakiFlow:

```toml
source-language = "auto"
target-language = "zh-CN"
output-dir = "./subtitles"
output-mode = "bilingual"

[agent]
backend = "codex"
```

Use `backend = "claude"` if you signed in with Claude Code. You can also omit
this file and pass the equivalent options on every run.

### 4. Download the default model and check the setup

```console
yakiflow models fetch
yakiflow doctor -c agent.backend=codex
```

Replace `codex` with `claude` when appropriate. Fix every reported `FAIL` that
applies to your input; `yt-dlp` is only needed for URLs, and the whisper.cpp
checks only apply to the Whisper backends. `models fetch` is unnecessary for
the ElevenLabs backends. A `WARN` line is a missing extra that no run depends
on, such as `tmux` for the review preview.

`yakiflow run` and `yakiflow resume` repeat these checks on startup and refuse
to start when one of them fails, so a setup that cannot finish is reported in
seconds instead of halfway through a job. `--skip-health-check` starts anyway.

### 5. Create subtitles

```console
yakiflow run video.mp4
```

Without `yakiflow.toml`, provide the three required settings explicitly:

```console
yakiflow run video.mp4 \
  --source-language auto \
  --target-language zh-CN \
  -c agent.backend=codex \
  --output-dir ./subtitles
```

YakiFlow shows transcription and translation progress. In an interactive
terminal, it then opens the selected Agent for final review. Tell the Agent any
changes you want, then end the review session when you are satisfied. The final
subtitle path is printed when the job finishes.

## Command-line options

A few high-frequency settings have dedicated options:

| Option | Setting |
| --- | --- |
| `-s`, `--source-language` | Source language code, or `auto` |
| `-t`, `--target-language` | Translation language or locale |
| `--transcription-backend` | One of the four transcription backends |
| `--stream` / `--no-stream` | Treat a URL as a live stream |
| `--output-dir` | Directory for final subtitle files |
| `--output-mode` | `source`, `translated`, `bilingual`, or `all` |
| `--context-file` | Copy a reference file into the work directory; repeatable |
| `--work-dir` | Place resumable job state in a chosen directory |
| `--keep-workdir` | Keep intermediate files after success |
| `--skip-health-check` | Start `run` or `resume` without the startup check |
| `--config-file` | Use this project configuration file |
| `--profile` | Select a `[profiles.NAME]` section |

Every other setting is changed with the repeatable `-c KEY=VALUE` option,
where `KEY` is the setting's dotted TOML key:

```console
yakiflow run video.mp4 \
  -c transcription.backend=elevenlabs \
  -c agent.draft.effort=medium \
  -c 'commands.yt-dlp-options=["--cookies-from-browser", "chrome"]'
```

Values are parsed as TOML (booleans, numbers, arrays); anything that does not
parse as TOML is taken as a plain string. Dedicated options override `-c`, and
`-c` overrides every configuration file.

## Transcription backends

Choose with `transcription.backend` (or `--transcription-backend`):

| Backend | What it is | Needs |
| --- | --- | --- |
| `whisper-cli` | Local whisper.cpp batch transcription (default) | `whisper-cli`, a model file |
| `whisper-server` | A whisper.cpp server — local, or an already-running one via `whisper.server-url` | `whisper-server`, or a reachable server |
| `elevenlabs` | ElevenLabs Scribe batch API, with speaker diarization | `yakiflow[elevenlabs]`, an API key |
| `elevenlabs-stream` | ElevenLabs Scribe realtime API over WebSocket | `yakiflow[elevenlabs]`, an API key |
| `fun-asr` | Fun-ASR asynchronous file HTTP API | `yakiflow[fun-asr]`, API key |
| `fun-asr-stream` | DashScope Recognition realtime SDK (WebSocket) | `yakiflow[fun-asr]`, API key |
| `mai` | MAI-Transcribe REST multipart API | `yakiflow[mai]`, Speech key and region/endpoint |
| `mai-stream` | MAI-Transcribe Streaming Speech SDK | `yakiflow[mai]`, Speech key and region |

Normal runs use `transcription.backend`; live capture uses `transcription.stream-backend`. For
live streams, `whisper-server` and `elevenlabs-stream` give the fastest live
preview; `whisper-cli` also works but reloads its model for every chunk, which
makes the preview lag well behind the stream.

The `elevenlabs` backend recognizes speakers (`elevenlabs.diarize`, on by
default): each subtitle's speaker label lands in the ASS `Name` field, and
overlapping speech becomes overlapping subtitles. When someone is interrupted,
their subtitle keeps running underneath the interrupting one instead of being
cut short at that moment. The realtime API does not diarize. With the ElevenLabs backends, subtitle line breaks and translations
are produced together by the draft Agent from the word-level transcript.

To use an already-running whisper-server instead of starting one:

```toml
stream = false           # --stream / --no-stream

[transcription]
# stream-backend = "whisper-server"  # auto: Whisper → whisper-server; ElevenLabs → elevenlabs-stream
backend = "whisper-server"

[whisper]
stream-chunk-seconds = 15
stream-context-seconds = 5
server-url = "http://127.0.0.1:8080"
```

### The ElevenLabs API key

The key is looked up in this order:

1. an `elevenlabs.api-key*` setting given with `-c` on the command line;
2. the `ELEVENLABS_API_KEY` environment variable;
3. an `elevenlabs.api-key*` setting from a configuration file;
4. the OS keyring (requires `yakiflow[keyring]`).

In a configuration file (or with `-c`), pick exactly one of three spellings:

```toml
[elevenlabs]
api-key = "sk-…"                  # the key itself
# api-key-file = "~/.secrets/elevenlabs"   # read the key from a file
# api-key-command = "pass show elevenlabs" # run a command; stdout is the key
```

To store the key in the OS keyring instead of any file:

```console
yakiflow secret set elevenlabs
yakiflow secret unset elevenlabs
```

`yakiflow doctor` reports where the key would come from without printing it.

## Common tasks

### Process a local file

Any audio or video format supported by `ffmpeg` can be used:

```console
yakiflow run ./recordings/interview.mkv \
  --source-language en \
  --target-language zh-CN \
  --output-dir ./subtitles
```

### Process a web video

```console
yakiflow run 'https://example.com/watch?v=123' \
  --source-language ja \
  --target-language en \
  -c download-dir=./downloads \
  --output-dir ./subtitles
```

`download-dir` keeps the downloaded media; omit it to use a temporary copy.
Set `--output-dir` for URL jobs so the subtitle location is predictable.
Playlists are not downloaded.

### Process a live stream

```console
yakiflow run 'https://example.com/live' \
  --stream \
  --source-language auto \
  --target-language zh-CN \
  --output-dir ./subtitles
```

After capture ends, choose review using live results, retranscription of the
complete recording, or exit preserving the work directory. In the TUI,
press `s` to stop capture and finalize what has been received; press Ctrl-C
once in a non-TUI run. While the stream is still running, press `o` to open
the recording so far with the current live subtitles in the media player.

The player loads the subtitle file only once. To make mpv pick up live
subtitle updates automatically, copy `contrib/mpv/sub-autoreload.lua` from
this repository into mpv's scripts directory (usually
`~/.config/mpv/scripts/`), or add
`--script=/path/to/contrib/mpv/sub-autoreload.lua` to
`review.video-open-command`. Setting that option replaces the default
command entirely, so start from it:

```toml
[review]
video-open-command = "mpv --script=/path/to/contrib/mpv/sub-autoreload.lua --no-sub-auto --no-resume-playback --sub-file={subtitle} --sid=auto --sub-visibility=yes {file}"
```

### Choose the subtitle format

Use `--output-mode` or set `output-mode` in TOML:

| Value | Result for an input named `video.mp4` |
| --- | --- |
| `bilingual` | `video.en-zh-cn.ass`, with translation above source text (default) |
| `source` | `video.source.ass` |
| `translated` | `video.translated.ass` |
| `all` | All three variants |

With automatic language detection, the bilingual filename uses the detected
language, for example `video.ja-zh-cn.ass`. Players such as `mpv` load ASS
files directly (`mpv --sub-file=video.en-zh-cn.ass video.mp4`).

Subtitles are written for the resolution of the input video, so text keeps its
shape and size on vertical, 4:3, and high-resolution sources alike. Without
`ffprobe`, or for an input with no picture, they are written for a 1920x1080
frame instead.

### Give each speaker its own style in Aegisub

With a diarizing backend, every subtitle carries its speaker in the ASS `Name`
(actor) field, while all lines share the `Default` style. To give speakers
different colors or positions in Aegisub:

1. Open the subtitle file in Aegisub and create one style per speaker in
   `Subtitle → Styles Manager`.
2. For each speaker, select any one of their lines and pick the new style in
   the edit box's style dropdown.
3. Load `contrib/aegisub/yakiflow-actor-styles.lua` from this repository via
   `Automation → Automation... → Add`, then run `Automation → Fill actor
   styles`.

Every remaining `Default` line takes the style you assigned to that speaker's
line. Lines you already restyled are left untouched, and one Ctrl-Z undoes the
whole run.

### Resume an interrupted job

In the TUI, press `s` to stop the current job cleanly and Ctrl-Q to leave the
interface. In a non-TUI run, the first Ctrl-C stops acquisition cleanly and a
second Ctrl-C forces exit. After an interruption or failure, YakiFlow prints
the preserved work directory. Resume it with:

```console
yakiflow resume /path/to/workdir
```

Keep the entire work directory intact. `resume` uses the original input and
settings; single settings can still be adjusted with `-c` when needed. A work
directory created by an older YakiFlow with a different configuration schema
must be finished by the version that created it. To choose the work directory
location in advance, start the job with `--work-dir ./work/video-job`.

## Configuration guide

Settings are read in this order, from highest to lowest priority:

1. dedicated command-line options;
2. `-c KEY=VALUE` overrides;
3. the selected profile in the file passed to `--config-file`, or in
   `yakiflow.toml` in the current directory;
4. the selected profile in the user configuration file;
5. base settings in the project configuration file;
6. base settings in the user configuration file;
7. built-in defaults.

On Linux, the user configuration file is
`~/.config/yakiflow/config.toml` (or
`$XDG_CONFIG_HOME/yakiflow/config.toml`). Relative paths are resolved from the
directory in which you start the job.

Settings are grouped into TOML tables. A full example with the most useful
keys:

```toml
# Top-level settings
source-language = "auto"
target-language = "zh-CN"
output-mode = "bilingual"        # source | translated | bilingual | all
output-dir = "./subtitles"
# download-dir = "./downloads"   # keep media downloaded from URLs
# memory = "./memory.md"         # persistent terminology/style memory file
# context-files = ["notes.txt"]  # read-only references for interactive review
# work-dir = "./work"            # resumable job state location
# keep-workdir = false

[transcription]
# stream-backend = "whisper-server"  # auto: Whisper → whisper-server; ElevenLabs → elevenlabs-stream
backend = "whisper-cli"   # whisper-cli | whisper-server | elevenlabs | elevenlabs-stream

[whisper]                 # used by whisper-cli and whisper-server
# model = "/path/to/ggml-large-v3-turbo-q5_0.bin"
# vad-model = "/path/to/ggml-silero-v6.2.0.bin"
# vad = true              # set false to ignore the configured VAD model
# cli = "whisper-cli"     # executable name or path
# server = "whisper-server"
# server-url = "http://127.0.0.1:8080"  # connect instead of launching

[elevenlabs]              # used by elevenlabs and elevenlabs-stream
max-minutes = 600
chunk-overlap-seconds = 30
# api-key-command = "pass show elevenlabs"
# model = "scribe_v2"
# realtime-model = "scribe_v2_realtime"
# diarize = true          # speaker recognition (batch API only)
# num-speakers = 3        # optional speaker-count hint
# use-speaker-library = true  # match speakers against the workspace library

[alignment]
# backend = "vad"         # vad | whisperx | none
# device = "auto"         # auto | cpu | cuda (WhisperX)
# model = "custom/model"  # override the WhisperX alignment model


[subtitles]
# max-cue-seconds = 8.0   # longest single subtitle (ElevenLabs backends)
# max-cue-chars = 84      # longest subtitle text per language (CJK counts double)

[agent]                   # shared by both translation stages
backend = "codex"         # codex | claude
# model = "…"             # rarely needed
# effort = "…"            # minimal | low | medium | high | xhigh
# extra-options = ["…"]   # extra argv tokens for the agent CLI

[agent.draft]             # batch draft translation; overrides [agent]
# backend = "…"
# model = "…"             # default: codex → gpt-5.6-terra, claude → sonnet
# effort = "low"
# extra-options = ["…"]
# workers = 4             # parallel draft requests
# batch-size = 20         # cues per request (Whisper backends)
# preceding-context = 10  # earlier cues shown for context
# following-context = 5   # later cues shown for context
# word-batch-size = 400   # words per request (ElevenLabs backends)
# word-following-context = 40
# timeout-seconds = 600
# max-attempts = 3
# retry-delay-seconds = 1

[agent.stream]            # inherits every agent.draft field; backend changes select its draft model
# batch-size = 5

[agent.final]             # interactive review; overrides [agent]
# backend = "…"
# model = "…"             # default: codex → gpt-5.6-sol, claude → opus
# effort = "high"
# extra-options = ["…"]

[review]
# display-mode = "split"  # split | open | both
# open-command = "code --reuse-window {workdir} {subtitle} {memory}"
# auto-open-video = false
# video-open-command = "vlc --sub-file={subtitle} {file}"

[commands]
# ffmpeg = "ffmpeg"
# ffprobe = "ffprobe"
# yt-dlp = "yt-dlp"
# yt-dlp-options = ["--cookies-from-browser", "chrome"]
```

The default Codex models are `gpt-5.6-terra` for draft translation and
`gpt-5.6-sol` for review. The Claude defaults are `sonnet` and `opus`. Override
them if those names are unavailable to your account. Settings in `[agent.draft]`
and `[agent.final]` win over the shared `[agent]` values, so the two stages can
use different backends.

`extra-options` and `yt-dlp-options` must be TOML arrays containing only
strings. YakiFlow passes each string directly as one argv token; options with a
value use two array items.

### Configuration profiles

Put reusable variants under `[profiles.NAME]` and select one with `--profile`
on `run`, `resume`, `doctor`, or `models fetch`:

```toml
source-language = "auto"
target-language = "zh-CN"

[agent]
backend = "codex"

[profiles.careful]
agent.draft.effort = "medium"

```

```console
yakiflow run input.mp4 --profile careful
yakiflow doctor --profile careful
```

Profiles may use dotted keys or nested tables and merge setting by setting: a
profile that changes `agent.draft.batch-size` keeps every other `[agent.draft]`
value from the base configuration. Arrays replace earlier arrays rather than
being appended. If both user and project files define the selected name, the
project profile wins; a profile is applied after both base files, and `-c` plus
dedicated options still win over profiles. Selecting an unknown profile is an
error, profiles cannot inherit from other profiles, and unselected profiles
have no effect. YakiFlow saves the fully resolved settings with each job, so a
resume does not depend on the profile or configuration files still existing.

### Reference files for review

Pass a danmaku export and a speaker note to the interactive review Agent as
read-only references:

```console
yakiflow run video.mp4 --context-file danmaku.xml --context-file speakers.txt
```

YakiFlow snapshots each file under `context/` in the job's work directory, so a
resumed job does not depend on the original reference files.

## Review and translation memory

When run from a terminal, YakiFlow starts an interactive review after batch
translation. The Agent checks the complete subtitle file, applies agreed edits,
and can propose reusable names, terminology, or style preferences. Memory is
only updated after you approve a proposal.

The review Agent writes to you in the target language and switches if you write
in another one. When the target language cannot keep one sentence split the way
the source was, it may merge those cues, mirroring the merge across every
configured artifact. Overlapping subtitles from different speakers are normal
with a diarizing backend and are kept as they are.

By default YakiFlow uses `tmux` to show a read-only subtitle pane next to the
Agent when possible. In that pane, press `o` to open the media with the current
subtitles in `mpv`. To use another player:

```toml
[review]
video-open-command = "vlc --sub-file={subtitle} {file}"
```

To open the source video automatically as the interactive review starts, enable
`review.auto-open-video`. The configured `video-open-command` is used, and the
staged subtitles are attached when the player supports them.

To open the staged subtitle file in an editor instead of using the split
preview:

```toml
[review]
display-mode = "open"
open-command = "code --reuse-window {workdir} {subtitle} {memory}"
```

Use `display-mode = "both"` to open the editor and retain the split preview at
the same time. `open-command` supports the `{subtitle}`, `{workdir}`, and
`{memory}` placeholders. Leave placeholders unquoted; YakiFlow safely keeps
substituted paths, including paths containing spaces, as single arguments. The
template uses shell-style argument parsing, but the resulting command is
launched directly without a shell.

In CI or with redirected input/output, interactive review is skipped and the
batch result is finalized directly.

## Optional: improve subtitle timing

These options apply to the Whisper backends. The ElevenLabs backends deliver
word-accurate timing already and default to `alignment.backend = "none"`, which
still holds each subtitle a moment past its last word instead of cutting it
there.

### VAD

A Silero VAD model helps Whisper skip silence and improves cue starts. Download
a model supported by your whisper.cpp build, then configure its path:

```toml
[whisper]
stream-chunk-seconds = 15
stream-context-seconds = 5
vad-model = "/absolute/path/to/ggml-silero-v6.2.0.bin"
```

`yakiflow models fetch` does not download VAD models. The
[whisper.cpp VAD guide](https://github.com/ggml-org/whisper.cpp#voice-activity-detection-vad)
has the model download commands.

Pass `-c whisper.vad=false` to run without VAD while keeping `vad-model` in the
config file. VAD is off whenever no model is configured.

### WhisperX forced alignment

WhisperX is optional and only adjusts timing; transcription still uses
whisper.cpp. Python 3.12 or 3.13 is required for the current extra. Install
exactly one variant with uv:

```console
# CPU
uv sync --extra whisperx-cpu

# NVIDIA CUDA 12.8
uv sync --extra whisperx-cuda
```

Then enable and verify it:

```toml
[alignment]
backend = "whisperx"
device = "auto" # auto, cpu, or cuda
```

```console
yakiflow doctor -c alignment.backend=whisperx
```

The first run may download NLTK data and a language-specific alignment model.
If alignment setup fails interactively, YakiFlow lets you retry or fall back to
VAD; a non-interactive run preserves the job for later resumption.

`whisperx` may be used together with `elevenlabs.diarize`. Subtitles that
overlap another speaker's keep their original times, so people talking over
each other stay that way; every other subtitle is aligned as usual.

## Troubleshooting

- Start with `yakiflow doctor` using the same options as your run.
- If a run stops with `health check failed`, fix the listed items; each one
  would otherwise have stopped the job later. `--skip-health-check` starts a
  run whose check you disagree with.
- If the default Whisper model is missing, run `yakiflow models fetch`.
- If a custom model path is configured, YakiFlow expects that file to already
  exist and will not replace it.
- `yt-dlp` failures only affect URL inputs; whisper.cpp failures only affect
  the Whisper backends.
- If an ElevenLabs run stops with a key error, `yakiflow doctor` shows which
  source the key would come from.
- If review has no subtitle pane, install `tmux` or configure
  `review.display-mode = "open"`.
- Keep the printed work directory after a failure. It contains the job journal
  and process logs needed to diagnose or resume the run.

## Development

```console
uv sync --extra dev
uv run --extra dev pytest
```

Live capture uses `agent.stream`, inheriting each field from `agent.draft`.
After stopping, remaining audio and final service results are saved before a
completion dialog offers review with live timestamps, full-recording
retranscription with `agent.draft`, or exit with the recording and work directory
preserved. Review uses `agent.final`. Resume repeats an unanswered dialog;
a saved choice resumes its execution. Without an interactive terminal, the full
recording is retranscribed automatically. Retranscription checks its resources
before clearing live words, translations and agent coverage.

Settings schema 3 rejects the old `[stream]` table and does not migrate old
work directories; finish those with the version that created them. Whisper
chunk/context settings now live in `[whisper]` and only affect Whisper capture.

### Fun-ASR

```toml
[transcription]
backend = "fun-asr"       # auto stream-backend: fun-asr-stream

[fun-asr]
model = "fun-asr"
realtime-model = "fun-asr-realtime"
api-url = "https://dashscope.aliyuncs.com/api/v1"
realtime-url = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
max-minutes = 720
chunk-overlap-seconds = 30
diarize-max-minutes = 120
diarize-over-limit = "disable" # or "split"
# api-key = "..."         # or api-key-file / api-key-command (mutually exclusive)
```

Install `yakiflow[fun-asr]`. Keys resolve from CLI, `DASHSCOPE_API_KEY`,
configuration, then `yakiflow secret set fun-asr` (requires `[keyring]`). Doctor
checks the HTTP dependency for batch, and DashScope for realtime.
Batch uses [temporary OSS upload](https://help.aliyun.com/en/model-studio/get-temporary-file-url)
through HTTP, followed by asynchronous submission, polling and result download.
Upload metadata, task ID and raw responses are saved in the work directory;
resume polls the existing task and reuses downloaded results. Expired links or
failed tasks report an error without silently submitting another paid task.
Uploads must fit the returned policy and 1 GB temporary-storage limit; models
support requests up to 12 hours. Longer recordings are split automatically. Flash
models are unsupported. API URLs can use your workspace's regional domain.

With `diarize-over-limit = "disable"` (the default), batch diarization is enabled
only when the complete recording is **strictly shorter** than
`diarize-max-minutes`; `0` always disables it. `"split"` keeps diarization enabled
by splitting each request below that threshold. The default 120 follows
[the provider's recommendation](https://help.aliyun.com/zh/model-studio/fun-asr-recorded-speech-recognition-http-api).
The actual decision is persisted for translation, subtitles, review and resume.
Realtime does not diarize; intermediate sentences are previews, final sentences
are durable, and stopping waits for the SDK's last results. Batch words share
the word segmentation pipeline; realtime uses segment timestamps. Both default
to no additional alignment. Credentials are never printed by doctor.

### MAI

```toml
[transcription]
backend = "mai"          # auto stream-backend: mai-stream

[mai]
# region = "swedencentral" # or SPEECH_REGION
# endpoint = "https://YourResourceName.cognitiveservices.azure.com"
model = "MAI-Transcribe-2"
realtime-model = "MAI-Transcribe-2-Streaming"
max-minutes = 120
chunk-overlap-seconds = 30
diarize-max-minutes = 15
diarize-over-limit = "disable" # or "split"
# api-key = "..."         # or api-key-file / api-key-command (mutually exclusive)
```

Install `yakiflow[mai]` (Speech SDK >= 1.52 for realtime). Credentials use
CLI overrides, `SPEECH_KEY`, configuration, then `yakiflow secret set mai`
(requires `[keyring]`). Set `mai.region` or `SPEECH_REGION`; batch can instead
use a REST `mai.endpoint`, either a resource base URL or the full transcription
URL. Realtime uses the region's universal v2 WebSocket endpoint through the
[official Speech SDK](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-transcribe-2-streaming-speech-sdk).
Doctor checks the dependency and credentials for the selected phase.

[Batch REST](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-transcribe)
requests word timestamps with enhanced mode. Each request must be shorter than
2 hours and smaller than 250 MB; longer or larger recordings are split automatically.
With the default `diarize-over-limit = "disable"`, diarization is enabled only
when the complete recording is below `mai.diarize-max-minutes`, a conservative
preview default of 15 minutes; exactly 15 minutes disables it for every chunk.
`"split"` keeps diarization enabled and limits each chunk to that threshold.
`0` always disables it; threshold overrides must be nonnegative integers. The complete recording's
actual decision is saved with the raw request and response and reused on resume.
Both MAI backends default to no additional alignment. Realtime uses segment
start/duration timestamps, keeps intermediate results as previews, persists
final segments, and waits for the last results after closing PCM input.
Cancellation, NoMatch and service errors are handled through the callback
thread adapter. MAI realtime does not enable diarization.

### Batch duration limits and recovery

`elevenlabs`, `fun-asr` and `mai` share automatic file splitting. Configure
`max-minutes` and `chunk-overlap-seconds` in the backend's TOML table or with
`-c`, for example `-c mai.diarize-over-limit=split -c mai.max-minutes=60`.
`max-minutes` accepts finite positive numbers; overlap accepts finite nonnegative
numbers. Defaults follow the single-request duration ceilings documented by
[ElevenLabs](https://elevenlabs.io/docs/overview/capabilities/speech-to-text),
[Fun-ASR](https://help.aliyun.com/zh/model-studio/asr-model) and
[MAI](https://learn.microsoft.com/en-us/rest/api/speechtotext/transcriptions/transcribe?view=rest-speechtotext-2025-10-15):

| Backend | `max-minutes` | `chunk-overlap-seconds` |
| --- | ---: | ---: |
| `elevenlabs` | 600 | 30 |
| `fun-asr` | 720 | 30 |
| `mai` | 120 | 30 |

Reaching a duration ceiling triggers splitting. A single plan uses the smallest
of the configured duration, the provider's PCM file size budget, and the
positive diarization threshold when `diarize-over-limit = "split"` applies.
Each chunk, including overlap, stays strictly below the limit with a PCM frame
of margin at an exact boundary. Fun-ASR obtains the upload policy before the
first plan and also observes the 1 GB temporary upload cap; MAI checks 250 MB
per chunk. Overlap must be smaller than the usable chunk length, otherwise the
job fails before uploading. There is no separate diarization overlap setting.
With `"disable"`, the complete recording determines diarization for every chunk;
short chunks never re-enable it. `diarize-max-minutes = 0` always turns it off.

Requests run sequentially. The work directory retains the complete plan, raw
responses and normalized words per chunk. Resume reuses fixed boundaries,
skips completed chunks, and polls Fun-ASR tasks already submitted for that
chunk. Changes to request settings or input audio are rejected to prevent
mixing results; restore the original settings or start a new job. Existing
whole-file result caches and Fun-ASR task checkpoints remain usable for whole-file
plans. An existing whole-file result/task cannot be reused for a split plan.

Merging restores original timestamps and matches overlap words using normalized
text and nearby times, ignoring punctuation differences. Speaker IDs carry
across chunks only with at least two consistent, uniquely corresponding word
matches; insufficient evidence, ties or conflicting labels produce a new ID.
With zero overlap, no speaker continuity is inferred. Matching cannot guarantee
speaker identity across silence, transcription differences or ambiguous
simultaneous speech. Matched duplicates appear once; other words belong to the
side of the overlap midpoint containing their center, keeping full word times
and simultaneous speech from different speakers. ElevenLabs timing repair runs
after merging. Translation starts from the complete merged transcript.

These file limits do not change realtime session recovery. Batch retranscription
after a live capture uses the same backend configuration and chunking rules.
