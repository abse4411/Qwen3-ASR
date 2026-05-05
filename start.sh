#!/usr/bin/env bash
# ============================================================================
# Qwen3-ASR 一键安装与启动脚本 (Linux / macOS)
#
# 用法:
#   ./start.sh                                  # 默认: transformers 后端 + 0.6B + 强制对齐器(时间戳)
#   ./start.sh --backend vllm                   # 使用 vLLM 后端 (需 CUDA)
#   ./start.sh --asr-model Qwen/Qwen3-ASR-1.7B  # 切换到 1.7B 模型
#   ./start.sh --no-aligner                     # 关闭强制对齐器(不返回时间戳)
#   ./start.sh --port 7860                      # 自定义端口
#   ./start.sh --skip-install                   # 跳过依赖安装,直接启动
#   ./start.sh --skip-ffmpeg                    # 跳过 ffmpeg 自动安装(用于视频上传)
#   ./start.sh --reinstall                      # 删除现有 .venv 重新安装
#   ./start.sh --help                           # 查看帮助
#
# 依赖管理: uv (https://github.com/astral-sh/uv)
# Python 版本: 3.12
# ============================================================================

set -euo pipefail

# ---------- 颜色输出 ----------
if [[ -t 1 ]]; then
    C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'
    C_INFO=$'\033[36m'; C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'
else
    C_RESET=""; C_BOLD=""; C_INFO=""; C_OK=""; C_WARN=""; C_ERR=""
fi
log_info()  { echo "${C_INFO}[INFO]${C_RESET} $*"; }
log_ok()    { echo "${C_OK}[ OK ]${C_RESET} $*"; }
log_warn()  { echo "${C_WARN}[WARN]${C_RESET} $*"; }
log_error() { echo "${C_ERR}[ERR ]${C_RESET} $*" >&2; }

# ---------- 默认参数 ----------
PYTHON_VERSION="3.12"
BACKEND="transformers"
ASR_MODEL="Qwen/Qwen3-ASR-0.6B"
ALIGNER_MODEL="Qwen/Qwen3-ForcedAligner-0.6B"
WITH_ALIGNER=1     # 默认启用强制对齐器(支持时间戳输出)
HOST="0.0.0.0"
PORT="8000"
CUDA_DEVICES="0"
SKIP_INSTALL=0
SKIP_FFMPEG=0
REINSTALL=0
EXTRA_ARGS=()

# ---------- 解析参数 ----------
print_help() {
    sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)        BACKEND="$2"; shift 2 ;;
        --asr-model)      ASR_MODEL="$2"; shift 2 ;;
        --aligner-model)  ALIGNER_MODEL="$2"; WITH_ALIGNER=1; shift 2 ;;
        --with-aligner)   WITH_ALIGNER=1; shift ;;
        --no-aligner)     WITH_ALIGNER=0; shift ;;
        --host|--ip)      HOST="$2"; shift 2 ;;
        --port)           PORT="$2"; shift 2 ;;
        --cuda-devices)   CUDA_DEVICES="$2"; shift 2 ;;
        --python-version) PYTHON_VERSION="$2"; shift 2 ;;
        --skip-install)   SKIP_INSTALL=1; shift ;;
        --skip-ffmpeg)    SKIP_FFMPEG=1; shift ;;
        --reinstall)      REINSTALL=1; shift ;;
        -h|--help)        print_help; exit 0 ;;
        --)               shift; EXTRA_ARGS+=("$@"); break ;;
        *)                EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ---------- 切到脚本所在目录 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
log_info "工作目录: $SCRIPT_DIR"

# ---------- 模型缓存 & 镜像端点 ----------
# 默认把模型下载到当前项目目录下 ./models, 可通过环境变量 MODELS_DIR 覆盖
MODELS_DIR="${MODELS_DIR:-${SCRIPT_DIR}/models}"
mkdir -p "${MODELS_DIR}"

# HuggingFace 相关缓存全部指向项目内 models 目录
export HF_HOME="${HF_HOME:-${MODELS_DIR}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${MODELS_DIR}/hub}"
# ModelScope 缓存(qwen-omni-utils 等可能用到)
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${MODELS_DIR}/modelscope}"

# 默认使用国内镜像 hf-mirror.com, 用户可通过环境变量 HF_ENDPOINT 覆盖
# 例如: HF_ENDPOINT=https://huggingface.co ./start.sh
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ---------- 抑制第三方库无害警告 ----------
# nagisa 0.2.11 在 Python 3.12 下会触发 SyntaxWarning(无效转义),
# 这是上游问题且不影响功能, 此处过滤掉. 用户可通过 PYTHONWARNINGS 自行覆盖.
# 如需查看所有警告, 运行: PYTHONWARNINGS=default ./start.sh
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::SyntaxWarning}"

log_info "模型缓存目录 : ${MODELS_DIR}"
log_info "HF_ENDPOINT  : ${HF_ENDPOINT}"

# ---------- 检测系统 ----------
OS_NAME="$(uname -s)"
case "$OS_NAME" in
    Linux*)  PLATFORM="linux" ;;
    Darwin*) PLATFORM="macos" ;;
    *)       log_error "不支持的系统: $OS_NAME"; exit 1 ;;
esac
log_info "检测到系统: $PLATFORM ($(uname -m))"

# ---------- 安装 uv ----------
ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        log_ok "已检测到 uv: $(uv --version)"
        return
    fi

    log_warn "未检测到 uv,开始自动安装..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        log_error "未找到 curl 或 wget, 无法自动安装 uv. 请手动安装: https://github.com/astral-sh/uv"
        exit 1
    fi

    # uv 默认安装到 ~/.local/bin 或 ~/.cargo/bin
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    if ! command -v uv >/dev/null 2>&1; then
        log_error "uv 安装后仍无法找到, 请打开新终端或手动将 uv 加入 PATH 后重试"
        exit 1
    fi
    log_ok "uv 安装成功: $(uv --version)"
}

# ---------- 创建/复用虚拟环境 ----------
setup_venv() {
    if [[ "$REINSTALL" -eq 1 && -d ".venv" ]]; then
        log_warn "--reinstall 指定, 删除现有 .venv ..."
        rm -rf .venv
    fi

    if [[ ! -d ".venv" ]]; then
        log_info "使用 Python ${PYTHON_VERSION} 创建虚拟环境 .venv ..."
        uv venv --python "${PYTHON_VERSION}" .venv
    else
        log_ok "复用已有虚拟环境 .venv"
    fi
}

# ---------- 安装依赖 ----------
install_deps() {
    log_info "安装 qwen-asr 项目依赖 (editable, backend=${BACKEND}) ..."
    if [[ "$BACKEND" == "vllm" ]]; then
        # vLLM 仅在 Linux + CUDA 环境完整可用
        if [[ "$PLATFORM" != "linux" ]]; then
            log_warn "vLLM 后端官方仅完整支持 Linux+CUDA, 当前为 ${PLATFORM},可能安装失败"
        fi
        uv pip install -e ".[vllm]" -i https://mirrors.cloud.tencent.com/pypi/simple
    else
        uv pip install -e . -i https://mirrors.cloud.tencent.com/pypi/simple
    fi
    log_ok "依赖安装完成"
}

# ---------- 安装 FFmpeg (用于视频上传时抽取音频) ----------
install_ffmpeg() {
    if command -v ffmpeg >/dev/null 2>&1; then
        log_ok "已检测到 ffmpeg: $(ffmpeg -version 2>/dev/null | head -n1)"
        return 0
    fi

    if [[ "$SKIP_FFMPEG" -eq 1 ]]; then
        log_warn "未检测到 ffmpeg, 但已指定 --skip-ffmpeg, 跳过自动安装"
        log_warn "  视频上传功能将不可用. 手动安装请见: https://ffmpeg.org/download.html"
        return 0
    fi

    log_warn "未检测到 ffmpeg, 尝试自动安装(视频上传功能依赖)..."

    # 提供 sudo 兜底 (容器环境/root 用户可能没有 sudo)
    SUDO=""
    if [[ "$EUID" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    fi

    if [[ "$PLATFORM" == "macos" ]]; then
        if command -v brew >/dev/null 2>&1; then
            log_info "  使用 brew 安装 ffmpeg ..."
            brew install ffmpeg || {
                log_warn "brew 安装 ffmpeg 失败, 视频上传功能将不可用"
                log_warn "  请手动执行: brew install ffmpeg"
                return 0
            }
        else
            log_warn "  未检测到 Homebrew. 请先安装 brew 再重试:"
            log_warn '    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
            log_warn "  之后执行: brew install ffmpeg"
            return 0
        fi
    elif [[ "$PLATFORM" == "linux" ]]; then
        if command -v apt-get >/dev/null 2>&1; then
            log_info "  使用 apt-get 安装 ffmpeg ..."
            $SUDO apt-get update -y >/dev/null 2>&1 || true
            $SUDO apt-get install -y ffmpeg || {
                log_warn "apt-get 安装失败, 请手动: $SUDO apt-get install -y ffmpeg"
                return 0
            }
        elif command -v dnf >/dev/null 2>&1; then
            log_info "  使用 dnf 安装 ffmpeg ..."
            $SUDO dnf install -y ffmpeg || {
                log_warn "dnf 安装失败, 请手动: $SUDO dnf install -y ffmpeg"
                return 0
            }
        elif command -v yum >/dev/null 2>&1; then
            log_info "  使用 yum 安装 ffmpeg ..."
            $SUDO yum install -y ffmpeg || {
                log_warn "yum 安装失败, 请手动: $SUDO yum install -y ffmpeg"
                return 0
            }
        elif command -v pacman >/dev/null 2>&1; then
            log_info "  使用 pacman 安装 ffmpeg ..."
            $SUDO pacman -Sy --noconfirm ffmpeg || {
                log_warn "pacman 安装失败, 请手动: $SUDO pacman -S ffmpeg"
                return 0
            }
        elif command -v zypper >/dev/null 2>&1; then
            log_info "  使用 zypper 安装 ffmpeg ..."
            $SUDO zypper install -y ffmpeg || {
                log_warn "zypper 安装失败, 请手动: $SUDO zypper install -y ffmpeg"
                return 0
            }
        else
            log_warn "  未识别的 Linux 发行版包管理器, 请手动安装 ffmpeg 后重试"
            return 0
        fi
    fi

    if command -v ffmpeg >/dev/null 2>&1; then
        log_ok "ffmpeg 安装成功: $(ffmpeg -version 2>/dev/null | head -n1)"
    else
        log_warn "ffmpeg 安装后仍无法找到, 视频上传功能将不可用"
    fi
}

# ---------- 探测当前可用的推理设备 ----------
# 输出: cuda / mps / cpu
detect_device() {
    uv run --no-sync python - <<'PY' 2>/dev/null || echo cpu
import sys
try:
    import torch
    if torch.cuda.is_available():
        print("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        print("mps")
    else:
        print("cpu")
except Exception:
    print("cpu")
PY
}

# ---------- 启动 WebUI ----------
launch_webui() {
    # 自动探测设备 (transformers 后端使用; vLLM 后端只支持 CUDA)
    DEVICE="$(detect_device | tail -n1 | tr -d '[:space:]')"
    [[ -z "$DEVICE" ]] && DEVICE="cpu"

    # 根据设备组装 backend-kwargs / aligner-kwargs 默认值
    AUTO_BACKEND_KWARGS=""
    AUTO_ALIGNER_KWARGS=""
    PASS_CUDA_DEVICES=1

    if [[ "$BACKEND" == "vllm" ]]; then
        # vLLM 仅 CUDA, 默认 kwargs 已合理, 不再自动注入 device_map
        if [[ "$DEVICE" != "cuda" ]]; then
            log_error "vLLM 后端仅支持 NVIDIA CUDA GPU, 但当前未检测到可用 CUDA 设备 (detected=${DEVICE})"
            log_error "请改用 --backend transformers, 或在带 NVIDIA GPU 的 Linux 主机上运行"
            exit 1
        fi
    else
        # transformers 后端: 根据设备选择 device_map / dtype
        case "$DEVICE" in
            cuda)
                AUTO_BACKEND_KWARGS='{"device_map":"cuda:0","dtype":"bfloat16","max_inference_batch_size":4,"max_new_tokens":4096}'
                AUTO_ALIGNER_KWARGS='{"device_map":"cuda:0","dtype":"bfloat16"}'
                ;;
            mps)
                # Apple Silicon: bfloat16 在 MPS 部分算子支持有限, 用 float16 更稳
                AUTO_BACKEND_KWARGS='{"device_map":"mps","dtype":"float16","max_inference_batch_size":1,"max_new_tokens":4096}'
                AUTO_ALIGNER_KWARGS='{"device_map":"mps","dtype":"float16"}'
                PASS_CUDA_DEVICES=0
                ;;
            cpu|*)
                AUTO_BACKEND_KWARGS='{"device_map":"cpu","dtype":"float32","max_inference_batch_size":1,"max_new_tokens":4096}'
                AUTO_ALIGNER_KWARGS='{"device_map":"cpu","dtype":"float32"}'
                PASS_CUDA_DEVICES=0
                ;;
        esac
    fi

    # 检查用户是否已通过 EXTRA_ARGS 自定义 --backend-kwargs / --aligner-kwargs
    USER_HAS_BACKEND_KWARGS=0
    USER_HAS_ALIGNER_KWARGS=0
    for a in "${EXTRA_ARGS[@]:-}"; do
        case "$a" in
            --backend-kwargs|--backend-kwargs=*) USER_HAS_BACKEND_KWARGS=1 ;;
            --aligner-kwargs|--aligner-kwargs=*) USER_HAS_ALIGNER_KWARGS=1 ;;
        esac
    done

    log_info "启动 Qwen3-ASR Gradio WebUI ..."
    log_info "  ASR 模型     : ${ASR_MODEL}"
    log_info "  后端         : ${BACKEND}"
    log_info "  推理设备     : ${DEVICE}"
    log_info "  监听地址     : ${HOST}:${PORT}"
    [[ "$PASS_CUDA_DEVICES" -eq 1 ]] && log_info "  CUDA 设备    : ${CUDA_DEVICES}"
    log_info "  模型缓存     : ${MODELS_DIR}"
    log_info "  HF_ENDPOINT  : ${HF_ENDPOINT}"
    [[ "$WITH_ALIGNER" -eq 1 ]] && log_info "  ForcedAligner: ${ALIGNER_MODEL}"

    # 组装命令
    cmd=(uv run --no-sync qwen-asr-demo
        --asr-checkpoint "${ASR_MODEL}"
        --backend "${BACKEND}"
        --ip "${HOST}"
        --port "${PORT}"
    )
    if [[ "$PASS_CUDA_DEVICES" -eq 1 ]]; then
        cmd+=(--cuda-visible-devices "${CUDA_DEVICES}")
    fi
    if [[ -n "$AUTO_BACKEND_KWARGS" && "$USER_HAS_BACKEND_KWARGS" -eq 0 ]]; then
        cmd+=(--backend-kwargs "$AUTO_BACKEND_KWARGS")
    fi
    if [[ "$WITH_ALIGNER" -eq 1 ]]; then
        cmd+=(--aligner-checkpoint "${ALIGNER_MODEL}")
        if [[ -n "$AUTO_ALIGNER_KWARGS" && "$USER_HAS_ALIGNER_KWARGS" -eq 0 ]]; then
            cmd+=(--aligner-kwargs "$AUTO_ALIGNER_KWARGS")
        fi
    fi
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    echo
    log_info "执行: ${cmd[*]}"
    echo
    exec "${cmd[@]}"
}

# ---------- 主流程 ----------
ensure_uv

if [[ "$SKIP_INSTALL" -eq 0 ]]; then
    setup_venv
    install_deps
else
    log_warn "已指定 --skip-install,跳过环境创建与依赖安装"
    if [[ ! -d ".venv" ]]; then
        log_error "虚拟环境 .venv 不存在,无法跳过安装. 请先运行一次完整安装."
        exit 1
    fi
fi

install_ffmpeg

launch_webui
