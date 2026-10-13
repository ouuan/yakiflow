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
TOML 键。所有键见[完整配置参考](#完整配置参考)：

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
effort = "high"

[review]
auto-open-video = true
```

`[agent]` 设置同时用于翻译和审校。可在 `[agent.draft]`、`[agent.final]` 中
分别设置 `backend`、`model` 和 `effort`；直播翻译使用 `[agent.stream]`，采用独立
默认值，并从 `[agent]` 继承共享设置。账户无法使用默认模型时，通过这些设置
指定可用模型。

额外命令行参数写成字符串数组，选项与其值分别占一个元素：

```toml
[commands]
yt-dlp-options = ["--cookies-from-browser", "chrome"]
```

Agent 的额外参数同样可通过 `agent.extra-options` 或各阶段的 `extra-options`
设置。完整命令行帮助见 `yakiflow run --help`。

### 完整配置参考

以下表格列出了所有用户可配置选项。`-c` 使用完整点分键；配置文件中可把点后面的
键写在相应 TOML 表中，例如 `whisper.vad = false` 等价于 `[whisper]` 下的
`vad = false`。文档使用连字符命名，也接受下划线；未知键会报错。
省略选项时使用默认值；“未设置”表示省略该键，TOML 不支持 null。
配置文件中的字符串需加引号，布尔值写作 `true`/`false`，额外参数使用字符串数组。
路径支持 `~`，相对路径相对于启动 YakiFlow 的目录解析。

#### 通用设置

| 键 | 默认值 | 含义 / 可选值 |
| --- | --- | --- |
| `source-language` | 未设置；必填 | 源语言代码，或 `auto` 自动检测。 |
| `target-language` | 未设置；必填 | 翻译目标语言或区域，如 `en`、`zh-CN`。 |
| `output-mode` | `"bilingual"` | `source`、`translated`、`bilingual` 或 `all`；见[字幕形式](#选择字幕形式)。 |
| `download-dir` | 未设置 | 在此目录保留 URL 下载的媒体；省略时下载到任务工作目录。 |
| `output-dir` | 未设置 | 最终字幕目录。默认写在本地媒体或保留的下载媒体旁；临时下载则写在当前目录。 |
| `memory` | 用户配置目录中的 `memory.md` | 共享翻译记忆；Linux 上通常为 `~/.config/yakiflow/memory.md`。 |
| `context-files` | `[]` | 审校参考文件路径数组；文件会复制到任务中。 |
| `work-dir` | 临时目录 | 保存任务状态和中间文件。新任务显式指定的目录必须为空，成功后保留。 |
| `keep-workdir` | `false` | 成功后保留自动创建的工作目录；中断和失败的任务会保留以供续跑。 |
| `stream` | `false` | 将 URL 作为直播流处理；采集时使用 `transcription.stream-backend` 和 `agent.stream`。 |

#### 转录与 Whisper

| 键 | 默认值 | 含义 / 可选值 |
| --- | --- | --- |
| `transcription.backend` | `"whisper-cli"` | `whisper-cli`、`whisper-server`、`elevenlabs`、`elevenlabs-stream`、`fun-asr`、`fun-asr-stream`、`mai` 或 `mai-stream`。 |
| `transcription.stream-backend` | 根据 `transcription.backend` 推导 | Whisper 后端默认使用 `whisper-server`；云端后端默认使用对应的 `-stream` 后端。可选 `whisper-cli`、`whisper-server`、`elevenlabs-stream`、`fun-asr-stream`、`mai-stream`。 |
| `whisper.model` | 用户缓存目录中的 `models/ggml-large-v3-turbo-q5_0.bin` | Whisper 模型路径；Linux 上通常为 `~/.cache/yakiflow/models/ggml-large-v3-turbo-q5_0.bin`。 |
| `whisper.vad` | `true` | 提供 `whisper.vad-model` 时启用 Whisper VAD。 |
| `whisper.vad-model` | 未设置 | Silero VAD 模型路径；没有模型时 Whisper VAD 关闭。 |
| `whisper.cli` | `"whisper-cli"` | Whisper CLI 可执行文件名称或路径。 |
| `whisper.server` | `"whisper-server"` | 启动本地 Whisper 服务时使用的可执行文件名称或路径。 |
| `whisper.server-url` | 未设置 | 已有服务的 HTTP(S) 推理 URL，包含接口路径；省略时启动本地服务。 |
| `whisper.stream-chunk-seconds` | `15` | 每个 Whisper 直播分块的新音频秒数；正整数。 |
| `whisper.stream-context-seconds` | `5` | Whisper 直播分块包含的前文音频秒数；非负整数。 |

#### 云端凭据

每个云端服务支持以下三种键。同一配置层中，每个服务最多选择一种形式；高优先级
配置层会整体替换低优先级的凭据选择。解析优先级为命令行 `-c`、环境变量、
配置文件、系统钥匙串；见[管理 API key](#管理-api-key)。

| 键 | 默认值 | 含义 |
| --- | --- | --- |
| `elevenlabs.api-key`、`fun-asr.api-key`、`mai.api-key` | 未设置 | API key 字符串。 |
| `elevenlabs.api-key-file`、`fun-asr.api-key-file`、`mai.api-key-file` | 未设置 | 包含 key 的 UTF-8 文件路径。 |
| `elevenlabs.api-key-command`、`fun-asr.api-key-command`、`mai.api-key-command` | 未设置 | 从指定命令的标准输出读取 key。 |

对应环境变量为 `ELEVENLABS_API_KEY`、`DASHSCOPE_API_KEY`、`SPEECH_KEY`。
未设置 `mai.region` 时还会读取 `SPEECH_REGION`。

#### ElevenLabs

| 键 | 默认值 | 含义 / 可选值 |
| --- | --- | --- |
| `elevenlabs.model` | `"scribe_v2"` | 文件转录模型 ID。 |
| `elevenlabs.realtime-model` | `"scribe_v2_realtime"` | 实时转录模型 ID。 |
| `elevenlabs.diarize` | `true` | 文件转录时识别说话人。 |
| `elevenlabs.num-speakers` | 未设置 | 预期说话人数，1 至 32 的整数；省略时自动估计。 |
| `elevenlabs.use-speaker-library` | `true` | 在识别说话人的文件转录中使用服务的说话人库。 |
| `elevenlabs.max-minutes` | `600` | 单次文件转录请求的最大分钟数；正数。 |
| `elevenlabs.chunk-overlap-seconds` | `30` | 文件转录分块之间的重叠秒数；非负且小于有效分块长度。 |

#### Fun-ASR

| 键 | 默认值 | 含义 / 可选值 |
| --- | --- | --- |
| `fun-asr.model` | `"fun-asr"` | 批量 Fun-ASR 模型 ID；此处不接受 Flash 或实时模型。 |
| `fun-asr.realtime-model` | `"fun-asr-realtime"` | 以 `fun-asr-realtime` 开头的实时模型 ID。 |
| `fun-asr.api-url` | `"https://dashscope.aliyuncs.com/api/v1"` | 文件转录的 HTTP(S) API 基础 URL。 |
| `fun-asr.realtime-url` | `"wss://dashscope.aliyuncs.com/api-ws/v1/inference"` | WebSocket（`ws`/`wss`）推理 URL。 |
| `fun-asr.diarize-max-minutes` | `120` | 非负整数分钟数；录音短于此时长时识别说话人，`0` 禁用识别。 |
| `fun-asr.diarize-over-limit` | `"disable"` | 录音达到或超过阈值时，`disable` 禁用说话人识别，或 `split` 拆分成识别说话人的分块。 |
| `fun-asr.max-minutes` | `720` | 单次文件转录请求的最大分钟数；正数。 |
| `fun-asr.chunk-overlap-seconds` | `30` | 文件转录分块之间的重叠秒数；非负且小于有效分块长度。 |

#### MAI

| 键 | 默认值 | 含义 / 可选值 |
| --- | --- | --- |
| `mai.region` | `SPEECH_REGION`，否则未设置 | Azure 区域标识。实时转录必填；文件转录可用 `mai.endpoint` 代替。 |
| `mai.endpoint` | 根据区域推导 | HTTP(S) 资源终结点或完整文件转录 URL；仅用于文件转录。 |
| `mai.model` | `"MAI-Transcribe-2"` | 文件转录模型 ID。 |
| `mai.realtime-model` | `"MAI-Transcribe-2-Streaming"` | 实时转录模型 ID。 |
| `mai.diarize-max-minutes` | `15` | 非负整数分钟数；录音短于此时长时识别说话人，`0` 禁用识别。 |
| `mai.diarize-over-limit` | `"disable"` | 录音达到或超过阈值时，`disable` 禁用说话人识别，或 `split` 拆分成识别说话人的分块。 |
| `mai.max-minutes` | `120` | 单次文件转录请求的最大分钟数；正数。 |
| `mai.chunk-overlap-seconds` | `30` | 文件转录分块之间的重叠秒数；非负且小于有效分块长度。 |

所有云端服务都可能因文件大小限制而提前拆分。说话人识别设置仅用于文件转录，
实时后端不识别说话人。使用 `split` 时，不同分块中的说话人标签可能不同。

#### 对齐与字幕

| 键 | 默认值 | 含义 / 可选值 |
| --- | --- | --- |
| `alignment.backend` | Whisper 为 `"vad"`，云端为 `"none"` | `vad`、`whisperx` 或 `none`；`vad` 需要 Whisper。直播采集期间始终为 `none`。 |
| `alignment.device` | `"auto"` | WhisperX 设备：`auto`、`cpu` 或 `cuda`。 |
| `alignment.model` | 未设置 | WhisperX 对齐模型名称；省略时根据检测到的源语言选择。 |
| `subtitles.max-cue-seconds` | `8.0` | 分段时每条字幕的最大秒数；正数。 |
| `subtitles.max-cue-chars` | `84` | 分段时每条字幕的字符数限制；至少为 8 的整数。按词分段时分别限制原文和译文，CJK 等宽字符按两个字符计。 |

按词分段时，若时长或字符数中一项小于其限制的一半，另一项可以超过限制。

#### Agent

以下四个键均可用于 `[agent]`、`[agent.draft]`、`[agent.final]`、
`[agent.stream]`。例如 `agent.backend` 设置共享后端，
`agent.final.backend` 仅覆盖审校后端。

| 表内的键 | 默认值 / 继承规则 | 含义 / 可选值 |
| --- | --- | --- |
| `backend` | `[agent]` 未设置；三个阶段均继承它 | `codex` 或 `claude`。当前翻译阶段（draft 或 stream）和 final 均必填，可共享设置或按阶段指定。 |
| `model` | `[agent]` 未设置；三个阶段均继承它，再使用下表中的阶段默认值 | Agent 模型名称。 |
| `effort` | `[agent]` 未设置；三个阶段均继承它，再使用下表中的阶段默认值 | `minimal`、`low`、`medium`、`high` 或 `xhigh`。 |
| `extra-options` | `[agent]` 未设置；三个阶段均继承它，否则为 `[]` | Agent CLI 额外参数的字符串数组；显式设置 `[]` 清除继承的参数。 |

| 阶段 | 默认 Codex 模型 | 默认 Claude 模型 | 默认 effort |
| --- | --- | --- | --- |
| `agent.draft` | `gpt-6-luna` | `sonnet` | `medium` |
| `agent.stream` | `gpt-6-luna` | `sonnet` | `low` |
| `agent.final` | `gpt-6.1-sol` | `opus` | `medium` |

各阶段独立解析共享项，优先级为：阶段设置 → `[agent]` → 阶段内置默认值。
`[agent.stream]` 不从 `[agent.draft]` 继承，调优选项使用下表中的 stream 默认值。
这些是 YakiFlow 内置的模型名称；可用 `model` 指定账户可用模型。

以下键仅用于 `[agent.draft]` 和 `[agent.stream]`，例如 `agent.draft.workers`
或 `agent.stream.workers`；不能放在 `[agent]` 或 `[agent.final]` 中。

| 表内的键 | Draft 默认值 | Stream 默认值 | 含义 / 可选值 |
| --- | --- | --- | --- |
| `workers` | `4` | `4` | 草稿 Agent 调用的最大并发数；正整数。 |
| `batch-size` | `20` | `3` | 基于字幕条目的转录每批翻译的条数；正整数。 |
| `preceding-context` | `10` | `6` | 作为上下文提供的前文字幕条数；非负整数，也用于按词分批。 |
| `following-context` | `5` | `0` | 按字幕条目分批时提供的后文原文字幕条数；非负整数。 |
| `word-batch-size` | `200` | `80` | 基于词的转录每批分段、翻译的目标词数；至少为 50 的整数。 |
| `word-following-context` | `40` | `0` | 按词分批时提供的后文原文词数；非负整数。 |
| `timeout-seconds` | `600` | `120` | 每次草稿 Agent 尝试的超时秒数；正数。 |
| `max-attempts` | `3` | `2` | 每批最大尝试次数，包含首次尝试；正整数。 |
| `retry-delay-seconds` | `1` | `1` | 失败后重试前的等待秒数；非负数。 |

Draft 默认值侧重吞吐和上下文，stream 默认值侧重小批次频繁输出和更短的失败恢复。
这些数值是调优起点，不保证固定延迟：直播的按词分批还需等待满足条件的静音间隔。

#### 审校与命令

| 键 | 默认值 | 含义 / 可选值 |
| --- | --- | --- |
| `review.display-mode` | `"split"` | `split` 在 tmux 可用时显示字幕窗格；`open` 启动外部查看器；`both` 同时使用两者。 |
| `review.open-command` | 未设置 | 字幕查看命令；`open` 或 `both` 模式必填。支持 `{subtitle}`（也可用 `{file}`、`{srt}`、`{srt_file}`）、`{workdir}`、`{memory}`（也可用 `{memory_file}`）；省略字幕占位符时追加字幕路径。 |
| `review.auto-open-video` | `false` | 交互审校开始时打开带字幕的源视频。 |
| `review.video-open-command` | 下方的 mpv 命令 | 媒体播放命令。支持 `{file}`（也可用 `{media}`）和 `{subtitle}`；省略占位符时追加媒体路径及兼容 mpv 的字幕选项。 |
| `commands.ffmpeg` | `"ffmpeg"` | ffmpeg 可执行文件名称或路径。 |
| `commands.ffprobe` | `"ffprobe"` | 用于读取媒体元数据的 ffprobe 可执行文件名称或路径。 |
| `commands.yt-dlp` | `"yt-dlp"` | 用于 URL 下载的 yt-dlp 可执行文件名称或路径。 |
| `commands.yt-dlp-options` | `[]` | yt-dlp 额外参数的字符串数组。 |

`review.video-open-command` 默认值为：

```text
mpv --no-sub-auto --no-resume-playback --sub-file={subtitle} --sid=auto --sub-visibility=yes {file}
```

字幕查看与媒体播放命令会拆分为参数后直接执行，不经过 shell。路径占位符会自动
加引号，不支持 shell 管道或重定向。

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
