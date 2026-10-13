# YakiFlow

简体中文 | [English](README.md)

YakiFlow 为本地视频、音频、网页视频和直播生成 ASS 字幕，支持原文、译文和
双语输出。你可以使用本地或云端转录，通过 Codex 或 Claude Code 翻译，并在
保存前与 Agent 一起审校。支持说话人识别的后端还能标注说话人并保留同时发言。

任务中断后可以续跑，无需从头开始。

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
视频预览。云端转录后端不需要 whisper.cpp 或本地模型，参见
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
云端转录后端不需要 `models fetch`。`WARN` 表示缺少可选工具，例如用于分屏
审校的 `tmux`。

`run` 和 `resume` 启动时也会检查环境。确认检查不适用时，可以加上
`--skip-health-check` 跳过。

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
| `--transcription-backend` | 选择下表中的转录后端 |
| `--stream` / `--no-stream` | 把 URL 作为直播流处理 |
| `--output-dir` | 最终字幕文件的目录 |
| `--output-mode` | `source`、`translated`、`bilingual` 或 `all` |
| `--context-file` | 为审校提供参考文件；可重复 |
| `--work-dir` | 指定任务工作目录，供中断后续跑 |
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

用 `--transcription-backend` 或配置中的 `transcription.backend` 选择：

| 后端 | 适用场景 | 需要 |
| --- | --- | --- |
| `whisper-cli` | 本地文件转录（默认） | `whisper-cli` 和模型文件 |
| `whisper-server` | 本地转录与直播预览 | `whisper-server` 和模型，或可访问的服务器 |
| `elevenlabs` | ElevenLabs Scribe 文件转录，支持说话人识别 | `.[elevenlabs]` 和 API key |
| `elevenlabs-stream` | ElevenLabs 实时转录 | `.[elevenlabs]` 和 API key |
| `fun-asr` | 阿里云 Fun-ASR 文件转录，支持说话人识别 | `.[fun-asr]` 和 API key |
| `fun-asr-stream` | 阿里云实时转录 | `.[fun-asr]` 和 API key |
| `mai` | Azure MAI 文件转录，支持说话人识别 | `.[mai]`、Speech key 和 region 或 endpoint |
| `mai-stream` | Azure MAI 实时转录 | `.[mai]`、Speech key 和 region |

直播默认使用所选服务的实时后端；Whisper 默认使用 `whisper-server`。需要单独
选择时设置 `transcription.stream-backend`。云端实时后端不支持说话人识别。

### Whisper

默认使用 `whisper-cli`，先运行 `yakiflow models fetch` 下载模型。
要连接已运行的 whisper-server：

```toml
[transcription]
backend = "whisper-server"

[whisper]
server-url = "http://127.0.0.1:8080"
```

### ElevenLabs

安装依赖、设置 key，再运行任务（以下示例沿用快速开始中的语言和 Agent 配置）：

```console
python -m pip install -e '.[elevenlabs]'
export ELEVENLABS_API_KEY='你的 API key'
yakiflow run video.mp4 --transcription-backend elevenlabs
```

文件转录默认识别说话人。可用 `-c elevenlabs.diarize=false` 关闭，或用
`-c elevenlabs.num-speakers=3` 提供说话人数提示。

### Fun-ASR

```console
python -m pip install -e '.[fun-asr]'
export DASHSCOPE_API_KEY='你的 API key'
yakiflow run video.mp4 --transcription-backend fun-asr
```

默认使用 `fun-asr` 和 `fun-asr-realtime` 模型，不支持 Flash 模型。业务空间
需要其他服务地址时，设置 `fun-asr.api-url` 和 `fun-asr.realtime-url`。
长录音的说话人识别设置见下文。

### MAI

```console
python -m pip install -e '.[mai]'
export SPEECH_KEY='你的 Speech key'
export SPEECH_REGION='你的资源区域'
yakiflow run video.mp4 --transcription-backend mai
```

区域也可通过 `mai.region` 配置。文件转录还可以用 `mai.endpoint` 指定资源
地址或完整转录 URL；实时转录必须提供区域。

### 管理 API key

三个云端服务都支持环境变量、配置文件和操作系统钥匙串。key 的优先级为：
命令行 `-c`、环境变量、配置文件、钥匙串。环境变量名称见上面的示例。

在配置文件中，以下三种写法只能选一种；使用其他服务时把表名改为 `fun-asr`
或 `mai`：

```toml
[elevenlabs]
api-key = "sk-…"
# api-key-file = "~/.secrets/elevenlabs"
# api-key-command = "pass show elevenlabs"
```

也可以安装 `.[keyring]`，用以下命令保存或删除 key；服务名同样可替换：

```console
yakiflow secret set elevenlabs
yakiflow secret unset elevenlabs
```

`yakiflow doctor` 会显示 key 的来源，但不会打印 key 本身。

### 长录音与说话人识别

云端文件转录会自动拆分长录音，无需手动裁剪。可在对应服务的配置表中调整
`max-minutes` 和 `chunk-overlap-seconds`：

| 配置表 | 默认单次转录时长上限（分钟） | 默认重叠时长（秒） |
| --- | ---: | ---: |
| `[elevenlabs]` | 600 | 30 |
| `[fun-asr]` | 720 | 30 |
| `[mai]` | 120 | 30 |

文件大小限制也可能使录音提前拆分。`max-minutes` 必须大于 0，重叠时长必须
非负且小于实际可用片段长度。

Fun-ASR 默认只对不足 120 分钟的完整录音启用说话人识别，MAI 则为不足
15 分钟；达到阈值后默认关闭。要在长录音中保留说话人识别，设置
`diarize-over-limit = "split"`，例如：

```toml
[mai]
diarize-over-limit = "split"
diarize-max-minutes = 15
```

`diarize-max-minutes` 接受非负整数分钟；设为 `0` 始终关闭说话人识别。
拆分后的录音中，同一个说话人可能获得不同编号，审校时请留意。

中断后用 `yakiflow resume` 复用已完成的转录。若提示输入或转录配置不兼容，
请恢复原配置或新建任务。

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

采集结束后，可选择使用实时结果进入审校、重新转录完整录像，或退出并保留工作
目录。没有可用的实时结果时，要先重新转录才能审校；非交互式运行会自动重新转录。
在 TUI 中按 `s` 停止采集并把已收到的内容收尾；非 TUI 运行按一次 Ctrl-C。
直播进行中可以按 `o`，在媒体播放器中打开目前已录到的部分并挂上当前的实时字幕。

要让 mpv 自动跟进实时字幕的更新，把本仓库的
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

日常配置只需写出要修改的设置。例如，让草稿翻译更仔细，并在审校时自动打开视频：

```toml
source-language = "auto"
target-language = "zh-CN"
output-mode = "bilingual"
output-dir = "./subtitles"
# memory = "./memory.md"  # 在这些任务之间共享术语与风格偏好

[agent]
backend = "codex"        # 或 claude

[agent.draft]
effort = "medium"

[review]
auto-open-video = true
```

`[agent]` 设置同时用于翻译和审校。可在 `[agent.draft]`、`[agent.final]` 中
分别设置 `backend`、`model` 和 `effort`；直播翻译使用 `[agent.stream]`，未设置的
选项沿用 `[agent.draft]`。账户无法使用默认模型时，通过这些设置指定可用模型。

额外命令行参数写成字符串数组，选项与其值分别占一个元素：

```toml
[commands]
yt-dlp-options = ["--cookies-from-browser", "chrome"]
```

Agent 的额外参数同样可通过 `agent.extra-options` 或各阶段的 `extra-options`
设置。完整命令行帮助见 `yakiflow run --help`。

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

profile 只覆盖其中写出的设置，其他值沿用基础配置；数组会整体替换。
项目与用户配置定义同名 profile 时，项目中的设置优先。续跑会保留原任务的配置，
无需保留原来的 profile 文件。

### 审校用的参考文件

把弹幕导出和说话人备注作为只读参考传给交互式审校 Agent：

```console
yakiflow run video.mp4 --context-file danmaku.xml --context-file speakers.txt
```

参考文件会随任务保存，续跑时无需再次提供。

## 审校与翻译记忆

在终端运行时，YakiFlow 会在批量翻译后启动交互式审校。Agent 检查完整的字幕
文件、应用商定的修改，并可以提议可复用的名称、术语或风格偏好。只有在你确认
后才会更新记忆。

可以直接告诉 Agent：“把这个人名统一译为……”“这句话太长，请重新断句”或
“后续任务也沿用这个术语”。Agent 默认用目标语言交流，也能跟随你切换语言。
通过顶层 `memory` 设置指定记忆文件，可在多个任务间共用术语和风格偏好。

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
`{subtitle}`、`{workdir}` 和 `{memory}` 占位符。占位符不要加引号，即使路径
包含空格也能使用。

在 CI 中或输入输出被重定向时，会跳过交互审校，直接完成批量结果。

## 可选：改善字幕时间轴

Whisper 用户可以尝试以下选项，改善字幕出现和消失的时间。云端转录通常无需
额外对齐。

### VAD

Silero VAD 模型能帮助 Whisper 跳过静音并改善字幕起始时间。下载你的
whisper.cpp 版本支持的模型，然后配置其路径：

```toml
[whisper]
vad-model = "/absolute/path/to/ggml-silero-v6.2.0.bin"
```

`yakiflow models fetch` 不会下载 VAD 模型。模型下载命令见
[whisper.cpp VAD 指南](https://github.com/ggml-org/whisper.cpp#voice-activity-detection-vad)。

传 `-c whisper.vad=false` 可以在保留配置文件中 `vad-model` 的同时禁用 VAD。
没有配置模型时 VAD 始终关闭。

### WhisperX 强制对齐

WhisperX 可以进一步改善字幕时间轴。需要 Python 3.12 或 3.13，用 uv 安装
两种变体之一：

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

## 疑难解答

- 先用与运行相同的选项执行 `yakiflow doctor`。
- 运行时若因 `health check failed` 停止，请逐项修复列出的内容；这些问题都会
  在任务中途导致失败。如果确定某项检查不适用，可以用 `--skip-health-check`
  直接开始。
- 缺少默认 Whisper 模型时，运行 `yakiflow models fetch`。
- 配置了自定义模型路径时，YakiFlow 要求该文件已经存在，不会替换它。
- `yt-dlp` 的故障只影响 URL 输入；whisper.cpp 的故障只影响 Whisper 后端。
- 云端任务因 key 报错停止时，`yakiflow doctor` 会显示 key 的来源。
- 审校没有字幕面板时，安装 `tmux` 或配置 `review.display-mode = "open"`。
- 失败后请保留打印出的工作目录。其中包含诊断或恢复运行所需的任务日志和进程
  日志。
