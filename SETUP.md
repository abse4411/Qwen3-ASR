# Qwen3-ASR 一键安装与启动指南

本仓库提供基于 [`uv`](https://github.com/astral-sh/uv) 的跨平台安装/启动脚本，
自动完成 **Python 3.12 环境创建 → 依赖安装 → WebUI 启动** 全流程。

支持平台：**Linux / macOS / Windows**。

---

## 1. 快速开始

### Linux / macOS

```bash
chmod +x start.sh
./start.sh
```

### Windows（PowerShell 或 CMD）

```bat
start.bat
```

> 首次运行会自动安装 `uv`（如本机未安装），然后创建 `.venv` 虚拟环境
> 并以可编辑模式安装本项目依赖，最后启动 Gradio WebUI。
> 启动成功后访问：<http://localhost:8000>

---

## 2. 常用命令

| 场景 | Linux/macOS | Windows |
|---|---|---|
| 默认启动（Transformers + 0.6B + 强制对齐器） | `./start.sh` | `start.bat` |
| 切换到 1.7B 模型 | `./start.sh --asr-model Qwen/Qwen3-ASR-1.7B` | `start.bat --asr-model Qwen/Qwen3-ASR-1.7B` |
| 关闭强制对齐器（不返回时间戳） | `./start.sh --no-aligner` | `start.bat --no-aligner` |
| 使用 vLLM 后端（需 CUDA） | `./start.sh --backend vllm` | `start.bat --backend vllm` |
| 自定义端口 | `./start.sh --port 7860` | `start.bat --port 7860` |
| 指定 GPU | `./start.sh --cuda-devices 1` | `start.bat --cuda-devices 1` |
| 跳过依赖安装直接启动 | `./start.sh --skip-install` | `start.bat --skip-install` |
| 跳过 ffmpeg 自动安装 | `./start.sh --skip-ffmpeg` | `start.bat --skip-ffmpeg` |
| 强制重建环境 | `./start.sh --reinstall` | `start.bat --reinstall` |
| 查看完整帮助 | `./start.sh --help` | `start.bat --help` |

任何 `start.sh / start.bat` 不识别的额外参数都会原样转发给底层的
`qwen-asr-demo` 命令，可追加例如：

```bash
./start.sh --with-aligner -- \
    --backend-kwargs '{"device_map":"cuda:0","dtype":"bfloat16","max_inference_batch_size":8}' \
    --aligner-kwargs '{"device_map":"cuda:0","dtype":"bfloat16"}'
```

---

## 3. 全部参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--backend`        | `transformers` | 推理后端，可选 `transformers` / `vllm` |
| `--asr-model`      | `Qwen/Qwen3-ASR-0.6B` | ASR 模型仓库 ID 或本地路径 |
| `--aligner-model`  | `Qwen/Qwen3-ForcedAligner-0.6B` | 强制对齐器仓库 ID 或本地路径 |
| `--with-aligner`   | **启用（默认）** | 启用强制对齐器以返回时间戳 |
| `--no-aligner`     | 关闭 | 关闭强制对齐器（仅 ASR、不返回时间戳） |
| `--host` / `--ip`  | `0.0.0.0` | 监听地址 |
| `--port`           | `8000` | 监听端口 |
| `--cuda-devices`   | `0` | `CUDA_VISIBLE_DEVICES`，CPU/Mac 可忽略 |
| `--python-version` | `3.12` | Python 版本，传给 `uv venv --python` |
| `--skip-install`   | 关闭 | 跳过虚拟环境创建与依赖安装 |
| `--skip-ffmpeg`    | 关闭 | 跳过 ffmpeg 自动安装（视频上传依赖此组件） |
| `--reinstall`      | 关闭 | 启动前删除现有 `.venv` 重建环境 |

---

## 4. 工作原理

1. **检测 / 安装 `uv`**
   - Linux/macOS：通过 `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows：通过 PowerShell `irm https://astral.sh/uv/install.ps1 | iex`
2. **创建虚拟环境**：`uv venv --python 3.12 .venv`
3. **安装依赖**：`uv pip install -e .`（vLLM 后端使用 `.[vllm]`）
4. **启动 WebUI**：`uv run --no-sync qwen-asr-demo ...`

整个过程不依赖 `conda`，不污染全局 Python，所有内容隔离在项目下的 `.venv` 中。

---

## 5. 模型缓存与下载镜像

启动脚本会**自动把所有模型下载到当前项目目录**，避免污染用户家目录、便于统一清理：

```
<项目根>/
├── .venv/        # uv 创建的 Python 3.12 环境
└── models/       # HuggingFace / ModelScope 模型缓存
    ├── hub/                # HF_HUB_CACHE
    └── modelscope/         # MODELSCOPE_CACHE
```

涉及的环境变量（脚本会自动设置，未显式配置时生效）：

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `MODELS_DIR`         | `<项目根>/models` | 模型根目录 |
| `HF_HOME`            | `${MODELS_DIR}` | HuggingFace 主缓存（含 `hub/`） |
| `HF_HUB_CACHE`       | `${MODELS_DIR}/hub` | `huggingface_hub` 模型缓存 |
| `MODELSCOPE_CACHE`   | `${MODELS_DIR}/modelscope` | ModelScope 缓存 |
| `HF_ENDPOINT`        | `https://hf-mirror.com` | **HuggingFace 镜像端点（默认走国内镜像）** |

### 切换镜像 / 直连官方

通过设置 `HF_ENDPOINT` 环境变量即可覆盖默认值：

```bash
# Linux/macOS — 直连 HuggingFace 官方
HF_ENDPOINT=https://huggingface.co ./start.sh

# 或永久导出
export HF_ENDPOINT=https://huggingface.co
./start.sh
```

```bat
:: Windows — 直连 HuggingFace 官方
set HF_ENDPOINT=https://huggingface.co
start.bat
```

### 自定义模型存放位置

```bash
# 例如把模型放到外接磁盘
MODELS_DIR=/Volumes/SSD/qwen-asr-models ./start.sh
```

### 使用本地已下载好的模型

直接通过 `--asr-model` 传入绝对路径即可，脚本不会再触发下载：

```bash
./start.sh --asr-model /abs/path/to/Qwen3-ASR-1.7B
```

---

## 6. 平台说明

启动脚本会**自动检测推理设备**（CUDA / Apple MPS / CPU）并按设备注入合适的 `device_map`、`dtype`：

| 检测到的设备 | `device_map` | `dtype`  | 备注 |
|---|---|---|---|
| `cuda`       | `cuda:0`     | `bfloat16` | NVIDIA GPU，性能最佳 |
| `mps`        | `mps`        | `float16`  | Apple Silicon (M1/M2/M3...) |
| `cpu`        | `cpu`        | `float32`  | 仅作可用性兜底，速度较慢 |

> 因此在 macOS / 无 NVIDIA 卡的机器上**无需任何额外参数**即可正常启动；
> 若用户通过 `-- --backend-kwargs '...'` 显式传入，脚本会**保留用户自定义值**，不再注入。

- **macOS**：仅支持 `transformers` 后端（vLLM 官方不支持 Apple Silicon/macOS）。
  自动选择 MPS，建议使用 0.6B 模型获得更好体验。
- **Linux + NVIDIA GPU**：全功能可用，推荐 `--backend vllm` 获得最高性能。
- **Windows**：
  - `transformers` 后端在 Windows 原生可用（CPU 或 CUDA）。
  - `vLLM` 在 Windows 原生支持有限，**强烈建议在 WSL2 中运行 `start.sh`**。
  - 选择了 `--backend vllm` 但未检测到 CUDA 时，脚本会直接报错退出，避免 `Torch not compiled with CUDA enabled`。

---

## 7. SRT 字幕导出

启用强制对齐器（默认开启）后，WebUI 在 **Timestamps（时间戳结果）** 面板下方
新增了一个 **Export SRT (导出字幕)** 按钮。流程：

1. 上传音频 / 视频 → 勾选 **Return Timestamps** → 点击 **Transcribe (识别)**。
2. 等待识别完成，**Result Text** 与 **Timestamps** 同时填充。
3. 点击 **Export SRT (导出字幕)** → 右侧 **Download SRT (下载字幕文件)**
   组件会出现一个可下载的 `.srt` 文件。

切分规则：

- 以 `Result Text` 中的标点符号为切分点，覆盖中英文常用标点
  （`，。！？；：、…—,.!?;:`），连续标点视为一处。
- 用 `Timestamps` 列表中的 `start_time` / `end_time` 作时间锚点：
  段时间 = 段内首 token 起始时间 → 段内末 token 结束时间。
- 输出标准 SRT 格式（`HH:MM:SS,mmm`），UTF-8 无 BOM，跨平台播放器
  （VLC、PotPlayer、IINA 等）通用。
- 文件名取上传源文件主名（`abc.mp4` → `abc.srt`）；纯录音/源名缺失时
  使用 `transcript.srt`。

> 若 `Result Text` 与 `Timestamps` 为空（未勾选时间戳），点击导出会提示
> 先开启 timestamps 并执行识别。

---

## 8. 视频上传 与 FFmpeg

WebUI 输入区拆为两个 Tab，**互斥使用**：

- **Audio Input (上传音频)**：直接上传音频文件。
- **Video Input (上传视频, 自动抽取音频)**：上传视频后，后端用系统 `ffmpeg`
  自动抽取 16kHz 单声道 WAV，再走识别流程；常见容器（`mp4` / `mov` / `mkv`
  / `webm` / `avi` …）均可直接使用。

### 8.1 自动安装 FFmpeg

启动脚本会在依赖安装之后自动尝试安装 `ffmpeg`：

| 平台 | 自动调用 |
|---|---|
| macOS  | `brew install ffmpeg`（需先安装 [Homebrew](https://brew.sh/)） |
| Linux  | 依次尝试 `apt-get` / `dnf` / `yum` / `pacman` / `zypper`（自动加 `sudo`） |
| Windows | 优先 `winget install --id Gyan.FFmpeg`，其次 `choco install ffmpeg -y` |

已安装 `ffmpeg` 时跳过；安装失败不会阻塞启动 —— 只是视频上传功能不可用，
脚本会打印对应的手动安装指引。

### 8.2 跳过自动安装

如果不需要视频功能、或希望自己管理 ffmpeg：

```bash
./start.sh --skip-ffmpeg
```

```bat
start.bat --skip-ffmpeg
```

### 8.3 手动安装

| 平台 | 命令 |
|---|---|
| macOS   | `brew install ffmpeg` |
| Ubuntu/Debian | `sudo apt-get install -y ffmpeg` |
| Fedora/RHEL   | `sudo dnf install -y ffmpeg`（CentOS 7 用 `yum`） |
| Arch    | `sudo pacman -S ffmpeg` |
| openSUSE | `sudo zypper install -y ffmpeg` |
| Windows (winget) | `winget install --id Gyan.FFmpeg` |
| Windows (choco)  | `choco install ffmpeg -y` |
| Windows (手动)   | 下载 https://www.gyan.dev/ffmpeg/builds/ 解压后将 `bin/` 加入 `PATH` |

> Windows 通过 winget/choco 安装后请**关闭并重新打开终端**让 `PATH` 生效，
> 否则 `start.bat` 仍可能找不到 `ffmpeg`。

---

## 9. 常见问题

**Q1. 提示 `uv: command not found`，且自动安装失败？**
手动安装：

- Linux/macOS：`curl -LsSf https://astral.sh/uv/install.sh | sh`
- Windows：`powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

完成后**重启终端**让 PATH 生效，再次运行启动脚本。

**Q2. 模型下载缓慢？**
建议先按 README 的 [Released Models Description and Download](README.md#released-models-description-and-download)
小节通过 `modelscope download ...` 把模型下载到本地，然后：

```bash
./start.sh --asr-model /abs/path/to/Qwen3-ASR-1.7B
```

**Q3. 8000 端口被占用？**
`./start.sh --port 7860`（或任何空闲端口）。

**Q4. 启动后浏览器麦克风权限报错？**
按 README 的 [HTTPS Notes](README.md#https-notes) 生成自签证书，然后：

```bash
./start.sh -- --ssl-certfile cert.pem --ssl-keyfile key.pem --no-ssl-verify
```
