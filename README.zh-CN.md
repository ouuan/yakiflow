# YakiFlow

简体中文 | [English](README.md)

YakiFlow 可以把本地视频、音频、网页视频或直播流转换成 ASS 字幕。它使用
`whisper.cpp` 或 ElevenLabs Scribe API 转录，通过 Codex 或 Claude Code 翻译，
并在保存前让你与交互式 Agent 一起审校结果。使用支持说话人识别的后端时，每条
字幕都带有说话人标签，两人同时说话会成为两条时间上重叠的字幕。

任务支持断点续跑：如果运行中断，YakiFlow 会保留已经完成的工作，并打印继续
任务所需的命令。

## 快速开始

### 1. 安装必需工具

你需要：

- Python 3.12 或更高版本；
- `ffmpeg`，以及随它一起安装的 `ffprobe`；
- `codex` 或 `claude` CLI，二选一安装并登录；
- 使用默认的本地转录时，还需要较新版本的
  [whisper.cpp](https://github.com/ggml-org/whisper.cpp)，且能在 `PATH` 中找到
  `whisper-cli`。

处理 URL 时还要安装 `yt-dlp`。`tmux` 和 `mpv` 是可选工具，分别用于分屏审校和
视频预览。ElevenLabs 后端不需要任何本地转录工具，参见
[转录后端](#转录后端)。

### 2. 安装 YakiFlow

在本仓库中运行：

```console
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

若要使用 ElevenLabs 后端，安装 `.[elevenlabs]`；若要把 API key 存进操作系统
钥匙串，再加上 `.[keyring]`：

```console
python -m pip install -e '.[elevenlabs,keyring]'
```

### 3. 保存常用设置

在准备运行 YakiFlow 的目录中创建 `yakiflow.toml`：

```toml
source-language = "auto"
target-language = "zh-CN"
output-dir = "./subtitles"
output-mode = "bilingual"

[agent]
backend = "codex"
```

如果使用 Claude Code，把后端改为 `backend = "claude"`。也可以不创建配置文件，
每次运行时都传入对应参数。

### 4. 下载默认模型并检查环境

```console
yakiflow models fetch
yakiflow doctor -c agent.backend=codex
```

使用 Claude Code 时把 `codex` 换成 `claude`。请修复所有与当前输入有关的
`FAIL`；`yt-dlp` 只影响 URL，whisper.cpp 相关检查只影响 Whisper 后端。
ElevenLabs 后端不需要 `models fetch`。标为 `WARN` 的是任何任务都不依赖的
可选项，例如用于预览字幕的 `tmux`。

`yakiflow run` 和 `yakiflow resume` 启动时会重跑这些检查，只要有一项失败就
拒绝开始，因此环境不满足时会立刻报错，而不是跑到一半才失败。加上
`--skip-health-check` 可以跳过检查直接开始。

### 5. 生成字幕

```console
yakiflow run video.mp4
```

没有 `yakiflow.toml` 时，显式提供三项必需设置：

```console
yakiflow run video.mp4 \
  --source-language auto \
  --target-language zh-CN \
  -c agent.backend=codex \
  --output-dir ./subtitles
```

YakiFlow 会显示转录和翻译进度。在交互式终端中，随后会打开所选的 Agent 进行
最终审校。把想要的修改告诉 Agent，满意后结束审校会话。任务完成时会打印最终
字幕文件的路径。

## 命令行选项

少数高频设置有专用选项：

| 选项 | 设置 |
| --- | --- |
| `-s`、`--source-language` | 源语言代码，或 `auto` |
| `-t`、`--target-language` | 目标语言或区域设置 |
| `--transcription-backend` | 四种转录后端之一 |
| `--stream` / `--no-stream` | 把 URL 作为直播流处理 |
| `--output-dir` | 最终字幕文件的目录 |
| `--output-mode` | `source`、`translated`、`bilingual` 或 `all` |
| `--context-file` | 把参考文件复制进工作目录；可重复 |
| `--work-dir` | 把可续跑的任务状态放到指定目录 |
| `--keep-workdir` | 成功后保留中间文件 |
| `--skip-health-check` | `run`、`resume` 启动时跳过环境检查 |
| `--config-file` | 使用指定的项目配置文件 |
| `--profile` | 选择一个 `[profiles.NAME]` 配置组 |

其余所有设置都通过可重复的 `-c KEY=VALUE` 选项修改，`KEY` 是该设置的点号
TOML 键：

```console
yakiflow run video.mp4 \
  -c transcription.backend=elevenlabs \
  -c agent.draft.effort=medium \
  -c 'commands.yt-dlp-options=["--cookies-from-browser", "chrome"]'
```

值按 TOML 解析（布尔、数字、数组）；无法按 TOML 解析的值当作普通字符串。
专用选项优先于 `-c`，`-c` 优先于所有配置文件。

## 转录后端

用 `transcription.backend`（或 `--transcription-backend`）选择：

| 后端 | 说明 | 需要 |
| --- | --- | --- |
| `whisper-cli` | 本地 whisper.cpp 批量转录（默认） | `whisper-cli` 和模型文件 |
| `whisper-server` | whisper.cpp 服务器——本地启动，或通过 `whisper.server-url` 连接已运行的实例 | `whisper-server`，或可访问的服务器 |
| `elevenlabs` | ElevenLabs Scribe 批量 API，支持说话人识别 | `yakiflow[elevenlabs]` 和 API key |
| `elevenlabs-stream` | ElevenLabs Scribe 实时 WebSocket API | `yakiflow[elevenlabs]` 和 API key |

四种后端都同时适用于普通任务和 `--stream` 直播采集。处理直播时，
`whisper-server` 和 `elevenlabs-stream` 的实时预览最快；`whisper-cli` 也可用，
但每个片段都要重新加载模型，预览会明显落后于直播。

`elevenlabs` 后端支持说话人识别（`elevenlabs.diarize`，默认开启）：每条字幕的
说话人标签写入 ASS 的 `Name` 字段，同时说话会成为时间上重叠的字幕。有人被打断
时，他的字幕会继续显示在插话的字幕下面，而不是就此中断。实时 API 不支持说话人
识别。使用 ElevenLabs 后端时，字幕的分行断句和翻译由 draft Agent
根据词级转录一并完成。

连接已运行的 whisper-server 而不是本地启动：

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

### ElevenLabs API key

key 按以下顺序查找：

1. 命令行 `-c` 传入的 `elevenlabs.api-key*` 设置；
2. 环境变量 `ELEVENLABS_API_KEY`；
3. 配置文件中的 `elevenlabs.api-key*` 设置；
4. 操作系统钥匙串（需要 `yakiflow[keyring]`）。

在配置文件（或 `-c`）中，三种写法只能选一种：

```toml
[elevenlabs]
api-key = "sk-…"                  # 直接写 key
# api-key-file = "~/.secrets/elevenlabs"   # 从文件读取 key
# api-key-command = "pass show elevenlabs" # 运行命令，stdout 是 key
```

把 key 存进操作系统钥匙串而不写任何文件：

```console
yakiflow secret set elevenlabs
yakiflow secret unset elevenlabs
```

`yakiflow doctor` 会报告 key 的来源，但不会打印 key 本身。

## 常见任务

### 处理本地文件

任何 `ffmpeg` 支持的音视频格式都可以使用：

```console
yakiflow run ./recordings/interview.mkv \
  --source-language en \
  --target-language zh-CN \
  --output-dir ./subtitles
```

### 处理网页视频

```console
yakiflow run 'https://example.com/watch?v=123' \
  --source-language ja \
  --target-language en \
  -c download-dir=./downloads \
  --output-dir ./subtitles
```

`download-dir` 会保留下载的媒体文件；省略则使用临时副本。处理 URL 时建议设置
`--output-dir`，让字幕位置可预期。不会下载播放列表。

### 处理直播流

```console
yakiflow run 'https://example.com/live' \
  --stream \
  --source-language auto \
  --target-language zh-CN \
  --output-dir ./subtitles
```

直播字幕是临时的。采集结束后，YakiFlow 会对完整录像重新转录，然后才发布最终
字幕。在 TUI 中按 `s` 停止采集并把已收到的内容收尾；非 TUI 运行按一次 Ctrl-C。
直播进行中可以按 `o`，在媒体播放器中打开目前已录到的部分并挂上当前的实时字幕。

播放器只会加载一次字幕文件。要让 mpv 自动跟进实时字幕的更新，把本仓库的
`contrib/mpv/sub-autoreload.lua` 复制到 mpv 的脚本目录（通常是
`~/.config/mpv/scripts/`），或在 `review.video-open-command` 中加上
`--script=/path/to/contrib/mpv/sub-autoreload.lua`。设置该选项会完全替换默认
命令，可以在默认命令的基础上修改：

```toml
[review]
video-open-command = "mpv --script=/path/to/contrib/mpv/sub-autoreload.lua --no-sub-auto --no-resume-playback --sub-file={subtitle} --sid=auto --sub-visibility=yes {file}"
```

### 选择字幕形式

使用 `--output-mode` 或在 TOML 中设置 `output-mode`：

| 值 | 输入名为 `video.mp4` 时的结果 |
| --- | --- |
| `bilingual` | `video.en-zh-cn.ass`，译文在上、原文在下（默认） |
| `source` | `video.source.ass` |
| `translated` | `video.translated.ass` |
| `all` | 以上三种全部输出 |

自动检测语言时，双语文件名使用检测到的语言，例如 `video.ja-zh-cn.ass`。`mpv`
等播放器可以直接加载 ASS 文件
（`mpv --sub-file=video.en-zh-cn.ass video.mp4`）。

字幕按输入视频的分辨率生成，因此在竖屏、4:3 和高分辨率片源上，文字的形状和大小
都保持正常。没有 `ffprobe`，或输入本身没有画面时，字幕按 1920x1080 生成。

### 在 Aegisub 中为每个说话人设置样式

使用带说话人识别的后端时，每条字幕的说话人都在 ASS 的 `Name`（说话人）字段中，
而所有行共用 `Default` 样式。要在 Aegisub 中给不同说话人不同的颜色或位置：

1. 在 Aegisub 中打开字幕文件，在「字幕 → 样式管理器」里为每个说话人建一个
   样式。
2. 对每个说话人，任选其一行，在编辑框的样式下拉菜单里选中新样式。
3. 通过「自动化 → 自动化... → 添加」加载本仓库的
   `contrib/aegisub/yakiflow-actor-styles.lua`，然后运行「自动化 → Fill actor
   styles」。

其余所有 `Default` 行都会改成该说话人那一行的样式。已经改过样式的行不会被
改动，整个操作一次 Ctrl-Z 即可撤销。

### 恢复中断的任务

在 TUI 中按 `s` 干净地停止当前任务，按 Ctrl-Q 退出界面。非 TUI 运行时，第一次
Ctrl-C 干净地停止采集，第二次 Ctrl-C 强制退出。中断或失败后，YakiFlow 会打印
保留的工作目录。用以下命令恢复：

```console
yakiflow resume /path/to/workdir
```

请保持整个工作目录完整。`resume` 使用原始输入和设置；必要时仍可用 `-c` 调整
单项设置。旧版 YakiFlow 用不同配置结构创建的工作目录，必须用创建它的版本
完成。要提前指定工作目录位置，启动任务时加 `--work-dir ./work/video-job`。

## 配置指南

设置按以下顺序读取，优先级从高到低：

1. 专用命令行选项；
2. `-c KEY=VALUE` 覆盖项；
3. `--config-file` 指定文件（或当前目录 `yakiflow.toml`）中被选中的 profile；
4. 用户配置文件中被选中的 profile；
5. 项目配置文件中的基础设置；
6. 用户配置文件中的基础设置；
7. 内置默认值。

在 Linux 上，用户配置文件是 `~/.config/yakiflow/config.toml`（或
`$XDG_CONFIG_HOME/yakiflow/config.toml`）。相对路径相对于启动任务的目录解析。

设置按 TOML 表分组。包含最常用键的完整示例：

```toml
# 顶层设置
source-language = "auto"
target-language = "zh-CN"
output-mode = "bilingual"        # source | translated | bilingual | all
output-dir = "./subtitles"
# download-dir = "./downloads"   # 保留从 URL 下载的媒体
# memory = "./memory.md"         # 持久的术语/风格记忆文件
# context-files = ["notes.txt"]  # 交互审校的只读参考文件
# work-dir = "./work"            # 可续跑任务状态的位置
# keep-workdir = false

[transcription]
# stream-backend = "whisper-server"  # auto: Whisper → whisper-server; ElevenLabs → elevenlabs-stream
backend = "whisper-cli"   # whisper-cli | whisper-server | elevenlabs | elevenlabs-stream

[whisper]                 # whisper-cli 与 whisper-server 共用
# model = "/path/to/ggml-large-v3-turbo-q5_0.bin"
# vad-model = "/path/to/ggml-silero-v6.2.0.bin"
# vad = true              # 设为 false 可忽略已配置的 VAD 模型
# cli = "whisper-cli"     # 可执行文件名或路径
# server = "whisper-server"
# server-url = "http://127.0.0.1:8080"  # 连接已运行实例而不本地启动

[elevenlabs]              # elevenlabs 与 elevenlabs-stream 共用
# api-key-command = "pass show elevenlabs"
# model = "scribe_v2"
# realtime-model = "scribe_v2_realtime"
# diarize = true          # 说话人识别（仅批量 API）
# num-speakers = 3        # 可选的说话人数提示
# use-speaker-library = true  # 与工作区的说话人库比对，识别已知说话人

[alignment]
# backend = "vad"         # vad | whisperx | none
# device = "auto"         # auto | cpu | cuda（WhisperX）
# model = "custom/model"  # 覆盖 WhisperX 对齐模型


[subtitles]
# max-cue-seconds = 8.0   # 单条字幕的最长时长（ElevenLabs 后端）
# max-cue-chars = 84      # 单条字幕单语言的最长字符数（CJK 按 2 个计）

[agent]                   # 两个翻译阶段共享
backend = "codex"         # codex | claude
# model = "…"             # 很少需要
# effort = "…"            # minimal | low | medium | high | xhigh
# extra-options = ["…"]   # 传给 Agent CLI 的额外 argv 参数

[agent.draft]             # 批量草稿翻译；覆盖 [agent]
# backend = "…"
# model = "…"             # 默认：codex → gpt-5.6-terra，claude → sonnet
# effort = "low"
# extra-options = ["…"]
# workers = 4             # 并行草稿请求数
# batch-size = 20         # 每次请求的字幕条数（Whisper 后端）
# preceding-context = 10  # 作为上下文的前文字幕条数
# following-context = 5   # 作为上下文的后文字幕条数
# word-batch-size = 400   # 每次请求的词数（ElevenLabs 后端）
# word-following-context = 40
# timeout-seconds = 600
# max-attempts = 3
# retry-delay-seconds = 1

[agent.stream]            # inherits every agent.draft field; backend changes select its draft model
# batch-size = 5

[agent.final]             # 交互审校；覆盖 [agent]
# backend = "…"
# model = "…"             # 默认：codex → gpt-5.6-sol，claude → opus
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

Codex 的默认模型是草稿翻译 `gpt-5.6-terra`、审校 `gpt-5.6-sol`；Claude 的默认
是 `sonnet` 和 `opus`。如果这些名称对你的账户不可用，请自行覆盖。
`[agent.draft]` 和 `[agent.final]` 中的设置优先于共享的 `[agent]` 值，因此两个
阶段可以使用不同的后端。

`extra-options` 和 `yt-dlp-options` 必须是只含字符串的 TOML 数组。YakiFlow 把
每个字符串直接作为一个 argv 参数传递；带值的选项要写成两个数组元素。

### 配置 profile

把可复用的变体放在 `[profiles.NAME]` 下，并在 `run`、`resume`、`doctor` 或
`models fetch` 时用 `--profile` 选择：

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

profile 可以使用点号键或嵌套表，并按单项设置合并：只修改
`agent.draft.batch-size` 的 profile 会保留基础配置中 `[agent.draft]` 的其他
值。数组整体替换而不是追加。用户和项目文件都定义了所选名称时，项目 profile
获胜；profile 在两个基础配置之后应用，而 `-c` 和专用选项仍然优先于 profile。
选择未知的 profile 是错误，profile 不能继承其他 profile，未被选中的 profile
没有任何影响。YakiFlow 会把完整解析后的设置随任务保存，因此续跑不依赖
profile 或配置文件仍然存在。

### 审校用的参考文件

把弹幕导出和说话人备注作为只读参考传给交互式审校 Agent：

```console
yakiflow run video.mp4 --context-file danmaku.xml --context-file speakers.txt
```

YakiFlow 会把每个文件快照到任务工作目录的 `context/` 下，因此续跑不依赖原始
参考文件。

## 审校与翻译记忆

在终端运行时，YakiFlow 会在批量翻译后启动交互式审校。Agent 检查完整的字幕
文件、应用商定的修改，并可以提议可复用的名称、术语或风格偏好。只有在你确认
后才会更新记忆。

审校 Agent 用目标语言与你交流；如果你换用其他语言，它也会跟着切换。当目标
语言无法保持原文的断句方式时，它可能合并这些字幕，并把合并同步到每个配置的
输出文件。使用支持说话人识别的后端时，不同说话人的字幕在时间上重叠是正常
现象，会原样保留。

默认情况下，YakiFlow 会在可能时用 `tmux` 在 Agent 旁边显示一个只读字幕面板。
在该面板中按 `o` 可用 `mpv` 打开带当前字幕的媒体。要换用其他播放器：

```toml
[review]
video-open-command = "vlc --sub-file={subtitle} {file}"
```

要在交互审校开始时自动打开源视频，启用 `review.auto-open-video`。会使用配置的
`video-open-command`，播放器支持时会挂载暂存的字幕。

要用编辑器打开暂存的字幕文件，而不是使用分屏预览：

```toml
[review]
display-mode = "open"
open-command = "code --reuse-window {workdir} {subtitle} {memory}"
```

`display-mode = "both"` 可以同时打开编辑器并保留分屏预览。`open-command` 支持
`{subtitle}`、`{workdir}` 和 `{memory}` 占位符。占位符不要加引号；YakiFlow 会
把替换后的路径（包括含空格的路径）安全地保持为单个参数。模板按 shell 风格
解析参数，但最终命令不经过 shell 直接启动。

在 CI 中或输入输出被重定向时，会跳过交互审校，直接完成批量结果。

## 可选：改善字幕时间轴

以下选项适用于 Whisper 后端。ElevenLabs 后端本身提供词级精度的时间轴，默认
`alignment.backend = "none"`；即使如此，每条字幕仍会在最后一个词说完后多留
一会儿，而不是立刻消失。

### VAD

Silero VAD 模型能帮助 Whisper 跳过静音并改善字幕起始时间。下载你的
whisper.cpp 版本支持的模型，然后配置其路径：

```toml
[whisper]
stream-chunk-seconds = 15
stream-context-seconds = 5
vad-model = "/absolute/path/to/ggml-silero-v6.2.0.bin"
```

`yakiflow models fetch` 不会下载 VAD 模型。模型下载命令见
[whisper.cpp VAD 指南](https://github.com/ggml-org/whisper.cpp#voice-activity-detection-vad)。

传 `-c whisper.vad=false` 可以在保留配置文件中 `vad-model` 的同时禁用 VAD。
没有配置模型时 VAD 始终关闭。

### WhisperX 强制对齐

WhisperX 是可选的，只调整时间轴；转录仍由 whisper.cpp 完成。当前 extra 需要
Python 3.12 或 3.13。用 uv 安装两种变体之一：

```console
# CPU
uv sync --extra whisperx-cpu

# NVIDIA CUDA 12.8
uv sync --extra whisperx-cuda
```

然后启用并验证：

```toml
[alignment]
backend = "whisperx"
device = "auto" # auto、cpu 或 cuda
```

```console
yakiflow doctor -c alignment.backend=whisperx
```

首次运行可能下载 NLTK 数据和语言专用的对齐模型。如果对齐初始化在交互式运行中
失败，YakiFlow 允许重试或退回 VAD；非交互式运行会保留任务供以后恢复。

`whisperx` 可以和 `elevenlabs.diarize` 同时使用。与另一位说话人重叠的字幕保留
原始时间，同时说话的效果得以保留；其余字幕照常对齐。

## 疑难解答

- 先用与运行相同的选项执行 `yakiflow doctor`。
- 运行时若因 `health check failed` 停止，请逐项修复列出的内容；这些问题都会
  在任务中途导致失败。如果确定某项检查不适用，可以用 `--skip-health-check`
  直接开始。
- 缺少默认 Whisper 模型时，运行 `yakiflow models fetch`。
- 配置了自定义模型路径时，YakiFlow 要求该文件已经存在，不会替换它。
- `yt-dlp` 的故障只影响 URL 输入；whisper.cpp 的故障只影响 Whisper 后端。
- ElevenLabs 任务因 key 报错停止时，`yakiflow doctor` 会显示 key 将来自哪个
  来源。
- 审校没有字幕面板时，安装 `tmux` 或配置 `review.display-mode = "open"`。
- 失败后请保留打印出的工作目录。其中包含诊断或恢复运行所需的任务日志和进程
  日志。

## 开发

```console
uv sync --extra dev
uv run --extra dev pytest
```

直播采集使用 `transcription.stream-backend` 和 `agent.stream`（逐项继承
`agent.draft`）。停止后先提交剩余音频并接收最终结果，再选择直接 review、
以非 stream 后端重转录完整录音、或退出保留录音和工作目录。无可用结果时禁用
直接 review。直接 review 保留实时时间戳；交互 review 使用 `agent.final`。
重转录使用 `transcription.backend`、`agent.draft` 与正常 alignment，资源检查
通过后清除实时词、翻译和覆盖状态。resume 恢复已选流程或再次询问；无 TTY
时自动重转录。配置 schema 3 拒绝旧 `[stream]`，不迁移旧工作目录。
