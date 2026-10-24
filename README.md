# YakiFlow

[简体中文](README.zh-CN.md) | English

YakiFlow creates ASS subtitles for local videos, audio files, web videos, and
live streams. Choose source text, translation, or bilingual output, use local
or cloud transcription, and translate with Codex, Claude Code, or Gemini CLI.
Before saving, you can review and edit the result with an interactive Agent.
Backends with speaker recognition label speakers and preserve simultaneous
speech.

Interrupted jobs can be resumed without starting over.

## Quick start

### 1. Install the required tools

You need:

- Python 3.12 or newer;
- `ffmpeg`, together with the `ffprobe` that ships with it;
- the `codex`, `claude`, or `gemini` CLI, installed and signed in;
- for the default local transcription, a recent
  [whisper.cpp](https://github.com/ggml-org/whisper.cpp) build whose
  `whisper-cli` executable is on `PATH`.

Install `yt-dlp` as well if you want to process URLs. `tmux` and `mpv` are
optional conveniences for subtitle review and video preview. Cloud backends
need neither whisper.cpp nor a local model — see
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

Use `backend = "claude"` if you signed in with Claude Code, or
`backend = "gemini"` for Gemini CLI. You can also omit this file and pass the
equivalent options on every run.

### 4. Download the default model and check the setup

```console
yakiflow models fetch
yakiflow doctor -c agent.backend=codex
```

Replace `codex` with `claude` or `gemini` when appropriate. Fix every reported
`FAIL` that applies to your input; `yt-dlp` is only needed for URLs, and the
whisper.cpp checks only apply to the Whisper backends. `models fetch` is
unnecessary for cloud backends. `WARN` indicates a missing optional tool, such
as `tmux` for split-pane review. `doctor` cannot tell whether Gemini CLI is
signed in, so run `gemini` once and sign in before your first job.

`run` and `resume` also check the setup on startup. Use `--skip-health-check`
if you are sure a reported check does not apply.

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
| `--transcription-backend` | Choose a transcription backend from the table below |
| `--stream` / `--no-stream` | Treat a URL as a live stream |
| `--output-dir` | Directory for final subtitle files |
| `--output-mode` | `source`, `translated`, `bilingual`, or `all` |
| `--context-file` | Provide a reference file for review; repeatable |
| `--work-dir` | Choose a job work directory for resuming after interruption |
| `--keep-workdir` | Keep intermediate files after success |
| `--skip-health-check` | Start `run` or `resume` without the startup check |
| `--config-file` | Use this project configuration file |
| `--profile` | Select a `[profiles.NAME]` section |

Every other setting is changed with the repeatable `-c KEY=VALUE` option,
where `KEY` is the setting's dotted TOML key. See the
[complete configuration reference](#complete-configuration-reference) for all keys:

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

Choose with `--transcription-backend` or `transcription.backend` in your config:

| Backend | Use it for | Needs |
| --- | --- | --- |
| `whisper-cli` | Local file transcription (default) | `whisper-cli`, a model file |
| `whisper-server` | Local transcription and live preview | `whisper-server` and a model, or a reachable server |
| `elevenlabs` | ElevenLabs Scribe file transcription with speaker recognition | `.[elevenlabs]`, an API key |
| `elevenlabs-stream` | ElevenLabs realtime transcription | `.[elevenlabs]`, an API key |
| `fun-asr` | Alibaba Cloud Fun-ASR file transcription with speaker recognition | `.[fun-asr]`, an API key |
| `fun-asr-stream` | Alibaba Cloud realtime transcription | `.[fun-asr]`, an API key |
| `mai` | Azure MAI file transcription with speaker recognition | `.[mai]`, a Speech key and region or endpoint |
| `mai-stream` | Azure MAI realtime transcription | `.[mai]`, a Speech key and region |

Live capture defaults to the selected service's realtime backend, or
`whisper-server` for Whisper. Set `transcription.stream-backend` to choose
separately. Cloud realtime backends do not recognize speakers.

### Whisper

The default is `whisper-cli`; run `yakiflow models fetch` to download its model.
To connect to an already-running whisper-server:

```toml
[transcription]
backend = "whisper-server"

[whisper]
server-url = "http://127.0.0.1:8080"
```

### ElevenLabs

Install the extra, set your key, and run a job. These examples use the language
and Agent settings from the quick start:

```console
python -m pip install -e '.[elevenlabs]'
export ELEVENLABS_API_KEY='your API key'
yakiflow run video.mp4 --transcription-backend elevenlabs
```

File transcription recognizes speakers by default. Disable it with
`-c elevenlabs.diarize=false`, or give a speaker-count hint with
`-c elevenlabs.num-speakers=3`.

### Fun-ASR

```console
python -m pip install -e '.[fun-asr]'
export DASHSCOPE_API_KEY='your API key'
yakiflow run video.mp4 --transcription-backend fun-asr
```

The default models are `fun-asr` and `fun-asr-realtime`; Flash models are not
supported. Set `fun-asr.api-url` and `fun-asr.realtime-url` if your workspace
requires different service addresses. See below for speaker recognition in
long recordings.

### MAI

```console
python -m pip install -e '.[mai]'
export SPEECH_KEY='your Speech key'
export SPEECH_REGION='your resource region'
yakiflow run video.mp4 --transcription-backend mai
```

You can also set the region with `mai.region`. For file transcription,
`mai.endpoint` can specify a resource address or full transcription URL instead.
Realtime transcription requires a region.

### Manage API keys

All three cloud services support environment variables, configuration files,
and the OS keyring. Key priority is: command-line `-c`, environment variable,
configuration file, then keyring. The examples above show the variable names.

In your config, choose exactly one of these three forms. Replace the table name
with `fun-asr` or `mai` for another service:

```toml
[elevenlabs]
api-key = "sk-…"
# api-key-file = "~/.secrets/elevenlabs"
# api-key-command = "pass show elevenlabs"
```

Alternatively, install `.[keyring]` and save or remove a key with these commands;
the service name can be changed in the same way:

```console
yakiflow secret set elevenlabs
yakiflow secret unset elevenlabs
```

`yakiflow doctor` shows the key's source without printing the key itself.

### Long recordings and speaker recognition

Cloud file transcription splits long recordings automatically; you do not need
to cut the media yourself. Adjust `max-minutes` and `chunk-overlap-seconds` in
the service's configuration table if needed:

| Configuration table | Default maximum minutes per request | Default overlap in seconds |
| --- | ---: | ---: |
| `[elevenlabs]` | 600 | 30 |
| `[fun-asr]` | 720 | 30 |
| `[mai]` | 120 | 30 |

File size limits may cause earlier splitting. `max-minutes` must be positive;
overlap must be nonnegative and shorter than the usable chunk length.

By default, Fun-ASR recognizes speakers only in complete recordings shorter
than 120 minutes, and MAI in recordings shorter than 15 minutes. At or above
the threshold, recognition is disabled. To retain it for long recordings, set
`diarize-over-limit = "split"`, for example:

```toml
[mai]
diarize-over-limit = "split"
diarize-max-minutes = 15
```

`diarize-max-minutes` accepts nonnegative whole minutes; `0` always disables
speaker recognition. The same speaker may receive different labels across
chunks, so check them during review.

After an interruption, use `yakiflow resume` to reuse completed transcription.
If it reports incompatible input or transcription settings, restore the original
settings or start a new job.

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
complete recording, or exit preserving the work directory. If no live results
are available, review requires retranscription. Non-interactive runs retranscribe
automatically. In the TUI, press `s` to stop capture and finalize what has been
received; press Ctrl-C
once in a non-TUI run. While the stream is still running, press `o` to open
the recording so far with the current live subtitles in the media player.

To make mpv pick up live subtitle updates automatically, copy
`contrib/mpv/sub-autoreload.lua` from this repository into mpv's scripts directory (usually
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

For everyday use, only include settings you want to change. For example, use
more reasoning for draft translation and open the video when review starts:

```toml
source-language = "auto"
target-language = "zh-CN"
output-mode = "bilingual"
output-dir = "./subtitles"
# memory = "./memory.md"  # share terminology and style between these jobs

[agent]
backend = "codex"        # or claude, gemini

[agent.draft]
effort = "high"

[review]
auto-open-video = true
```

`[agent]` settings apply to translation and review. Set `backend`, `model`, and
`effort` separately in `[agent.draft]` and `[agent.final]` to customize each.
Live translation uses `[agent.stream]`, with independent defaults and shared
settings from `[agent]`. If your account cannot use a default model, choose an
available model with these settings.

Write extra command-line arguments as string arrays, with an option and its
value in separate items:

```toml
[commands]
yt-dlp-options = ["--cookies-from-browser", "chrome"]
```

Agent arguments can likewise be set with `agent.extra-options` or a stage's
`extra-options`. For the full command-line help, run `yakiflow run --help`.

### Complete configuration reference

The tables below list every user-configurable setting. Use the dotted key with
`-c`, or put the key after the dot in its TOML table: `whisper.vad = false`
is equivalent to `vad = false` under `[whisper]`. Hyphenated names are the
documented spelling; underscores are also accepted. Unknown keys are errors.
Defaults apply when a setting is omitted; “unset” means omit the key, since
TOML has no null value. Strings must be quoted in TOML files, booleans use
`true`/`false`, and argument lists use arrays of strings. Paths expand `~`;
relative paths use the directory in which you start YakiFlow.

#### General settings

| Key | Default | Meaning / accepted values |
| --- | --- | --- |
| `source-language` | Unset; required | Source language code, or `auto` for detection. |
| `target-language` | Unset; required | Translation language or locale, such as `en` or `zh-CN`. |
| `output-mode` | `"bilingual"` | `source`, `translated`, `bilingual`, or `all`; see [subtitle formats](#choose-the-subtitle-format). |
| `download-dir` | Unset | Keep URL downloads in this directory; otherwise download into the job's work directory. |
| `output-dir` | Unset | Final subtitle directory. By default, write beside local or retained downloaded media; for temporary downloads, use the current directory. |
| `memory` | User config directory's `memory.md` | Shared translation memory; on Linux, normally `~/.config/yakiflow/memory.md`. |
| `context-files` | `[]` | Array of reference file paths copied into the job for review. |
| `work-dir` | Temporary directory | Job state and intermediate files. An explicitly chosen directory must be empty for a new job and is retained after success. |
| `keep-workdir` | `false` | Retain an automatically created work directory after success; interrupted and failed jobs are preserved for resumption. |
| `stream` | `false` | Treat a URL as a live stream; use `transcription.stream-backend` and `agent.stream` during capture. |

#### Transcription and Whisper

| Key | Default | Meaning / accepted values |
| --- | --- | --- |
| `transcription.backend` | `"whisper-cli"` | `whisper-cli`, `whisper-server`, `elevenlabs`, `elevenlabs-stream`, `fun-asr`, `fun-asr-stream`, `mai`, or `mai-stream`. |
| `transcription.stream-backend` | Derived from `transcription.backend` | Either Whisper backend defaults to `whisper-server`; cloud backends default to their corresponding `-stream` backend. Accepts `whisper-cli`, `whisper-server`, `elevenlabs-stream`, `fun-asr-stream`, or `mai-stream`. |
| `whisper.model` | User cache directory's `models/ggml-large-v3-turbo-q5_0.bin` | Whisper model path; on Linux, normally `~/.cache/yakiflow/models/ggml-large-v3-turbo-q5_0.bin`. |
| `whisper.vad` | `true` | Enable Whisper VAD when `whisper.vad-model` is provided. |
| `whisper.vad-model` | Unset | Silero VAD model path. Without a model, Whisper VAD is off. |
| `whisper.cli` | `"whisper-cli"` | Whisper CLI executable name or path. |
| `whisper.server` | `"whisper-server"` | Whisper server executable name or path, used when starting a local server. |
| `whisper.server-url` | Unset | Existing server's HTTP(S) inference URL, including its endpoint path; otherwise YakiFlow starts a local server. |
| `whisper.stream-chunk-seconds` | `15` | Positive integer seconds of new audio per Whisper live chunk. |
| `whisper.stream-context-seconds` | `5` | Nonnegative integer seconds of preceding audio included in each Whisper live chunk. |

#### Cloud credentials

Each cloud provider supports the following three keys. Choose at most one form
per provider in each configuration layer; a higher layer replaces the entire
credential choice. Resolution order is command-line `-c`, environment variable,
configuration file, then OS keyring; see [API key management](#manage-api-keys).

| Keys | Default | Meaning |
| --- | --- | --- |
| `elevenlabs.api-key`, `fun-asr.api-key`, `mai.api-key` | Unset | API key string. |
| `elevenlabs.api-key-file`, `fun-asr.api-key-file`, `mai.api-key-file` | Unset | Path to a UTF-8 file containing the key. |
| `elevenlabs.api-key-command`, `fun-asr.api-key-command`, `mai.api-key-command` | Unset | Command whose standard output supplies the key. |

The corresponding environment variables are `ELEVENLABS_API_KEY`,
`DASHSCOPE_API_KEY`, and `SPEECH_KEY`. `SPEECH_REGION` also supplies `mai.region`
when that setting is omitted.

#### ElevenLabs

| Key | Default | Meaning / accepted values |
| --- | --- | --- |
| `elevenlabs.model` | `"scribe_v2"` | File transcription model ID. |
| `elevenlabs.realtime-model` | `"scribe_v2_realtime"` | Realtime transcription model ID. |
| `elevenlabs.diarize` | `true` | Recognize speakers in file transcription. |
| `elevenlabs.num-speakers` | Unset | Expected speaker count, an integer from 1 to 32; omit for automatic estimation. |
| `elevenlabs.use-speaker-library` | `true` | Use the service's speaker library during diarized file transcription. |
| `elevenlabs.max-minutes` | `600` | Positive maximum minutes per file transcription request. |
| `elevenlabs.chunk-overlap-seconds` | `30` | Nonnegative overlap in seconds between file transcription chunks; must be shorter than the usable chunk length. |

#### Fun-ASR

| Key | Default | Meaning / accepted values |
| --- | --- | --- |
| `fun-asr.model` | `"fun-asr"` | Batch Fun-ASR model ID; Flash and realtime models are not accepted here. |
| `fun-asr.realtime-model` | `"fun-asr-realtime"` | Realtime model ID beginning with `fun-asr-realtime`. |
| `fun-asr.api-url` | `"https://dashscope.aliyuncs.com/api/v1"` | HTTP(S) API base URL for file transcription. |
| `fun-asr.realtime-url` | `"wss://dashscope.aliyuncs.com/api-ws/v1/inference"` | WebSocket (`ws`/`wss`) inference URL. |
| `fun-asr.diarize-max-minutes` | `120` | Nonnegative integer minutes. Recognize speakers below this recording duration; `0` disables recognition. |
| `fun-asr.diarize-over-limit` | `"disable"` | At or above the duration threshold, `disable` speaker recognition or `split` into diarized chunks. |
| `fun-asr.max-minutes` | `720` | Positive maximum minutes per file transcription request. |
| `fun-asr.chunk-overlap-seconds` | `30` | Nonnegative overlap in seconds between file transcription chunks; must be shorter than the usable chunk length. |

#### MAI

| Key | Default | Meaning / accepted values |
| --- | --- | --- |
| `mai.region` | `SPEECH_REGION`, otherwise unset | Azure region identifier. Required for realtime transcription, and for file transcription unless `mai.endpoint` is set. |
| `mai.endpoint` | Derived from region | HTTP(S) resource endpoint or full file transcription URL; applies to file transcription only. |
| `mai.model` | `"MAI-Transcribe-2"` | File transcription model ID. |
| `mai.realtime-model` | `"MAI-Transcribe-2-Streaming"` | Realtime transcription model ID. |
| `mai.diarize-max-minutes` | `15` | Nonnegative integer minutes. Recognize speakers below this recording duration; `0` disables recognition. |
| `mai.diarize-over-limit` | `"disable"` | At or above the duration threshold, `disable` speaker recognition or `split` into diarized chunks. |
| `mai.max-minutes` | `120` | Positive maximum minutes per file transcription request. |
| `mai.chunk-overlap-seconds` | `30` | Nonnegative overlap in seconds between file transcription chunks; must be shorter than the usable chunk length. |

File size limits may cause earlier splitting for any cloud provider. Speaker
recognition settings apply to file transcription; realtime backends do not
recognize speakers. With `split`, speaker labels may differ between chunks.

#### Alignment and subtitles

| Key | Default | Meaning / accepted values |
| --- | --- | --- |
| `alignment.backend` | `"vad"` for Whisper; `"none"` for cloud backends | `vad`, `whisperx`, or `none`. `vad` requires Whisper. During live capture, alignment is always `none`. |
| `alignment.device` | `"auto"` | WhisperX device: `auto`, `cpu`, or `cuda`. |
| `alignment.model` | Unset | WhisperX alignment model name; omit to select a model for the detected source language. |
| `subtitles.max-cue-seconds` | `8.0` | Positive maximum subtitle cue duration in seconds for segmentation. |
| `subtitles.max-cue-chars` | `84` | Character limit per segmented cue; integer of at least 8. In word-based segmentation, applies per language, with CJK and other wide characters counting as two. |

For word-based segmentation, either duration or character limit may be exceeded
when the other measure stays below half its limit.

#### Agents

These four keys are available in each of `[agent]`, `[agent.draft]`,
`[agent.final]`, and `[agent.stream]`. For example, use `agent.backend` for a
shared choice or `agent.final.backend` to override review only.

| Key within the table | Default / inheritance | Meaning / accepted values |
| --- | --- | --- |
| `backend` | Unset in `[agent]`; all three stages inherit it | `codex`, `claude`, or `gemini`. Required for the active translation stage (draft or stream) and final, either shared or set per stage. |
| `model` | Unset in `[agent]`; all three stages inherit it, then use the stage default below | Agent model name. |
| `effort` | Unset in `[agent]`; all three stages inherit it, then use the stage default below | `minimal`, `low`, `medium`, `high`, or `xhigh`. |
| `extra-options` | Unset in `[agent]`; all three stages inherit it, otherwise `[]` | Additional Agent CLI arguments as a string array; `[]` explicitly clears inherited arguments. |

| Stage | Default Codex model | Default Claude model | Default Gemini model | Default effort |
| --- | --- | --- | --- | --- |
| `agent.draft` | `gpt-6-luna` | `sonnet` | `flash` | `medium` |
| `agent.stream` | `gpt-6-luna` | `sonnet` | `flash` | `low` |
| `agent.final` | `gpt-6.1-sol` | `opus` | `pro` | `medium` |

Each stage resolves its shared keys independently: stage setting, then
`[agent]`, then its built-in default. `[agent.stream]` does not inherit from
`[agent.draft]`; its tuning keys use the stream defaults below.
These are YakiFlow's built-in model names; set `model` to a model your account
can access.

The following keys are available in `[agent.draft]` and `[agent.stream]`, for
example `agent.draft.workers` or `agent.stream.workers`. They are not accepted
in `[agent]` or `[agent.final]`.

| Key within the table | Draft default | Stream default | Meaning / accepted values |
| --- | --- | --- | --- |
| `workers` | `4` | `4` | Positive integer maximum concurrent draft Agent calls. |
| `batch-size` | `20` | `3` | Positive integer cues per translation batch for cue-based transcription. |
| `preceding-context` | `10` | `6` | Nonnegative integer preceding cues supplied as context; also used for word-based batches. |
| `following-context` | `5` | `0` | Nonnegative integer following source cues supplied as context for cue-based batches. |
| `word-batch-size` | `200` | `80` | Target words per segmentation/translation batch for word-based transcription; integer of at least 50. |
| `word-following-context` | `40` | `0` | Nonnegative integer following source words supplied as context for word-based batches. |
| `timeout-seconds` | `600` | `120` | Positive timeout in seconds for each draft Agent attempt. |
| `max-attempts` | `3` | `2` | Positive integer maximum attempts per batch, including the first attempt. |
| `retry-delay-seconds` | `1` | `1` | Nonnegative delay in seconds between failed attempts. |

Draft defaults favor throughput and context; stream defaults favor smaller,
more frequent batches and shorter failure recovery. These are starting points,
not a latency guarantee: live word batches also wait for a qualified silence.

#### Review and commands

| Key | Default | Meaning / accepted values |
| --- | --- | --- |
| `review.display-mode` | `"split"` | `split` shows a subtitle pane in tmux when available; `open` launches an external viewer; `both` does both. |
| `review.open-command` | Unset | Subtitle viewer command; required for `open` or `both`. Supports `{subtitle}` (also `{file}`, `{srt}`, `{srt_file}`), `{workdir}`, and `{memory}` (also `{memory_file}`); appends the subtitle path if its placeholder is omitted. |
| `review.auto-open-video` | `false` | Open the source video with subtitles when interactive review starts. |
| `review.video-open-command` | mpv command below | Media player command. Supports `{file}` (also `{media}`) and `{subtitle}`; missing placeholders cause the media path and an mpv-compatible subtitle option to be added. |
| `commands.ffmpeg` | `"ffmpeg"` | ffmpeg executable name or path. |
| `commands.ffprobe` | `"ffprobe"` | ffprobe executable name or path for media metadata. |
| `commands.yt-dlp` | `"yt-dlp"` | yt-dlp executable name or path for URL downloads. |
| `commands.yt-dlp-options` | `[]` | Extra yt-dlp arguments as a string array. |

The default `review.video-open-command` is:

```text
mpv --no-sub-auto --no-resume-playback --sub-file={subtitle} --sid=auto --sub-visibility=yes {file}
```

Review viewer and media player commands are split into arguments and run without
a shell. Path placeholders are quoted automatically; shell pipes and redirections
are not supported.

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

A profile overrides only the settings it contains; other values keep their
base settings. Arrays are replaced in full. When user and project files define
the same profile, project settings take priority. Resuming retains the original
job settings, so you do not need to keep the original profile file.

### Reference files for review

Pass a danmaku export and a speaker note to the interactive review Agent as
read-only references:

```console
yakiflow run video.mp4 --context-file danmaku.xml --context-file speakers.txt
```

Reference files are saved with the job; you do not need to provide them again
when resuming.

## Review and translation memory

When run from a terminal, YakiFlow starts an interactive review after batch
translation. The Agent checks the complete subtitle file, applies agreed edits,
and can propose reusable names, terminology, or style preferences. Memory is
only updated after you approve a proposal.

You can ask the Agent to use a particular spelling for a name, shorten a long
subtitle, or remember a term for future jobs. It communicates in the target
language by default and follows you if you switch languages. Set the top-level
`memory` path to share terminology and style preferences between jobs.

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
`{memory}` placeholders. Leave placeholders unquoted, including when paths
contain spaces.

In CI or with redirected input/output, interactive review is skipped and the
batch result is finalized directly.

## Optional: improve subtitle timing

Whisper users can try these options to improve when subtitles appear and
disappear. Cloud transcription usually needs no additional alignment.

### VAD

A Silero VAD model helps Whisper skip silence and improves cue starts. Download
a model supported by your whisper.cpp build, then configure its path:

```toml
[whisper]
vad-model = "/absolute/path/to/ggml-silero-v6.2.0.bin"
```

`yakiflow models fetch` does not download VAD models. The
[whisper.cpp VAD guide](https://github.com/ggml-org/whisper.cpp#voice-activity-detection-vad)
has the model download commands.

Pass `-c whisper.vad=false` to run without VAD while keeping `vad-model` in the
config file. VAD is off whenever no model is configured.

### WhisperX forced alignment

WhisperX can further improve subtitle timing. It requires Python 3.12 or 3.13.
Install exactly one variant with uv:

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
- If a cloud run stops with a key error, `yakiflow doctor` shows the key source.
- If review has no subtitle pane, install `tmux` or configure
  `review.display-mode = "open"`.
- Keep the printed work directory after a failure. It contains the job journal
  and process logs needed to diagnose or resume the run.
