@echo off
REM ============================================================================
REM Qwen3-ASR 一键安装与启动脚本 (Windows)
REM
REM 用法:
REM   start.bat                                   默认: transformers 后端 + 0.6B + 强制对齐器(时间戳)
REM   start.bat --backend vllm                    使用 vLLM 后端 (需 CUDA)
REM   start.bat --asr-model Qwen/Qwen3-ASR-1.7B   切换到 1.7B 模型
REM   start.bat --no-aligner                      关闭强制对齐器(不返回时间戳)
REM   start.bat --port 7860                       自定义端口
REM   start.bat --skip-install                    跳过依赖安装直接启动
REM   start.bat --skip-ffmpeg                     跳过 ffmpeg 自动安装(用于视频上传)
REM   start.bat --reinstall                       删除现有 .venv 重新安装
REM   start.bat --help                            查看帮助
REM
REM 依赖管理: uv (https://github.com/astral-sh/uv)
REM Python 版本: 3.12
REM ============================================================================
setlocal EnableDelayedExpansion

REM ---------- 默认参数 ----------
set "PYTHON_VERSION=3.12"
set "BACKEND=transformers"
set "ASR_MODEL=Qwen/Qwen3-ASR-0.6B"
set "ALIGNER_MODEL=Qwen/Qwen3-ForcedAligner-0.6B"
set "WITH_ALIGNER=1"
set "HOST=0.0.0.0"
set "PORT=8000"
set "CUDA_DEVICES=0"
set "SKIP_INSTALL=0"
set "SKIP_FFMPEG=0"
set "REINSTALL=0"
set "EXTRA_ARGS="

REM ---------- 解析参数 ----------
:parse_args
if "%~1"=="" goto :args_done
if /I "%~1"=="--backend"        ( set "BACKEND=%~2" & shift & shift & goto :parse_args )
if /I "%~1"=="--asr-model"      ( set "ASR_MODEL=%~2" & shift & shift & goto :parse_args )
if /I "%~1"=="--aligner-model"  ( set "ALIGNER_MODEL=%~2" & set "WITH_ALIGNER=1" & shift & shift & goto :parse_args )
if /I "%~1"=="--with-aligner"   ( set "WITH_ALIGNER=1" & shift & goto :parse_args )
if /I "%~1"=="--no-aligner"     ( set "WITH_ALIGNER=0" & shift & goto :parse_args )
if /I "%~1"=="--host"           ( set "HOST=%~2" & shift & shift & goto :parse_args )
if /I "%~1"=="--ip"             ( set "HOST=%~2" & shift & shift & goto :parse_args )
if /I "%~1"=="--port"           ( set "PORT=%~2" & shift & shift & goto :parse_args )
if /I "%~1"=="--cuda-devices"   ( set "CUDA_DEVICES=%~2" & shift & shift & goto :parse_args )
if /I "%~1"=="--python-version" ( set "PYTHON_VERSION=%~2" & shift & shift & goto :parse_args )
if /I "%~1"=="--skip-install"   ( set "SKIP_INSTALL=1" & shift & goto :parse_args )
if /I "%~1"=="--skip-ffmpeg"    ( set "SKIP_FFMPEG=1" & shift & goto :parse_args )
if /I "%~1"=="--reinstall"      ( set "REINSTALL=1" & shift & goto :parse_args )
if /I "%~1"=="--help"           ( goto :print_help )
if /I "%~1"=="-h"               ( goto :print_help )
set "EXTRA_ARGS=!EXTRA_ARGS! %~1"
shift
goto :parse_args

:print_help
echo.
echo Qwen3-ASR Windows 启动脚本
echo.
echo 选项:
echo   --backend ^<transformers^|vllm^>   选择推理后端 (默认 transformers)
echo   --asr-model ^<repo_or_path^>      ASR 模型 (默认 Qwen/Qwen3-ASR-0.6B)
echo   --aligner-model ^<repo_or_path^>  强制对齐器 (默认 Qwen/Qwen3-ForcedAligner-0.6B)
echo   --with-aligner                  启用强制对齐器(默认已启用)
echo   --no-aligner                    关闭强制对齐器(不返回时间戳)
echo   --host ^<ip^>                     监听 IP (默认 0.0.0.0)
echo   --port ^<port^>                   监听端口 (默认 8000)
echo   --cuda-devices ^<idx^>            CUDA_VISIBLE_DEVICES (默认 0)
echo   --python-version ^<x.y^>          Python 版本 (默认 3.12)
echo   --skip-install                  跳过依赖安装
echo   --skip-ffmpeg                   跳过 ffmpeg 自动安装(用于视频上传)
echo   --reinstall                     删除 .venv 重新安装
echo   --help                          打印此帮助
echo.
exit /b 0

:args_done

REM ---------- 切到脚本目录 ----------
cd /d "%~dp0"
echo [INFO] 工作目录: %CD%

REM ---------- 模型缓存 & 镜像端点 ----------
REM 默认把模型下载到当前项目目录下 .\models, 可通过环境变量 MODELS_DIR 覆盖
if not defined MODELS_DIR set "MODELS_DIR=%CD%\models"
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"

REM HuggingFace 相关缓存全部指向项目内 models 目录
if not defined HF_HOME            set "HF_HOME=%MODELS_DIR%"
if not defined HF_HUB_CACHE       set "HF_HUB_CACHE=%MODELS_DIR%\hub"
if not defined MODELSCOPE_CACHE   set "MODELSCOPE_CACHE=%MODELS_DIR%\modelscope"

REM 默认使用国内镜像 hf-mirror.com, 可通过环境变量 HF_ENDPOINT 覆盖
REM 例如: set HF_ENDPOINT=https://huggingface.co ^&^& start.bat
if not defined HF_ENDPOINT set "HF_ENDPOINT=https://hf-mirror.com"

REM ---------- 抑制第三方库无害警告 ----------
REM nagisa 0.2.11 在 Python 3.12 下会触发 SyntaxWarning(无效转义),
REM 上游问题且不影响功能, 此处过滤. 如需查看所有警告:
REM   set PYTHONWARNINGS=default ^&^& start.bat
if not defined PYTHONWARNINGS set "PYTHONWARNINGS=ignore::SyntaxWarning"

echo [INFO] 模型缓存目录 : %MODELS_DIR%
echo [INFO] HF_ENDPOINT  : %HF_ENDPOINT%

REM ---------- 检查 uv ----------
where uv >nul 2>&1
if errorlevel 1 (
    echo [WARN] 未检测到 uv,开始自动安装...
    where powershell >nul 2>&1
    if errorlevel 1 (
        echo [ERR ] 未找到 PowerShell,无法自动安装 uv
        echo        请手动安装 uv: https://github.com/astral-sh/uv
        exit /b 1
    )
    powershell -ExecutionPolicy ByPass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
    REM uv 默认安装到 %USERPROFILE%\.local\bin
    set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
    where uv >nul 2>&1
    if errorlevel 1 (
        echo [ERR ] uv 安装失败或未在 PATH, 请打开新终端再试
        exit /b 1
    )
)
for /f "delims=" %%V in ('uv --version') do echo [ OK ] 已检测到 %%V

REM ---------- 处理 --reinstall ----------
if "%REINSTALL%"=="1" (
    if exist .venv (
        echo [WARN] --reinstall 指定, 删除现有 .venv ...
        rmdir /s /q .venv
    )
)

REM ---------- 创建虚拟环境 + 安装依赖 ----------
if "%SKIP_INSTALL%"=="1" (
    if not exist .venv (
        echo [ERR ] .venv 不存在,无法 --skip-install,请先运行一次完整安装
        exit /b 1
    )
    echo [WARN] 已跳过依赖安装
) else (
    if not exist .venv (
        echo [INFO] 使用 Python %PYTHON_VERSION% 创建虚拟环境 .venv ...
        uv venv --python %PYTHON_VERSION% .venv
        if errorlevel 1 ( echo [ERR ] 创建虚拟环境失败 & exit /b 1 )
    ) else (
        echo [ OK ] 复用已有虚拟环境 .venv
    )

    echo [INFO] 安装 qwen-asr 项目依赖 (backend=%BACKEND%) ...
    if /I "%BACKEND%"=="vllm" (
        echo [WARN] vLLM 后端在 Windows 原生环境受限,如失败建议使用 WSL2/Linux
        uv pip install -e ".[vllm]" -i https://mirrors.cloud.tencent.com/pypi/simple
    ) else (
        uv pip install -e . -i https://mirrors.cloud.tencent.com/pypi/simple
    )
    if errorlevel 1 ( echo [ERR ] 依赖安装失败 & exit /b 1 )
    echo [ OK ] 依赖安装完成
)

REM ---------- 安装 FFmpeg (用于视频上传时抽取音频) ----------
where ffmpeg >nul 2>&1
if not errorlevel 1 (
    echo [ OK ] 已检测到 ffmpeg
) else (
    if "%SKIP_FFMPEG%"=="1" (
        echo [WARN] 未检测到 ffmpeg, 已指定 --skip-ffmpeg, 跳过自动安装
        echo        视频上传功能将不可用. 手动安装请见: https://www.gyan.dev/ffmpeg/builds/
    ) else (
        echo [WARN] 未检测到 ffmpeg, 尝试自动安装(视频上传功能依赖)...
        where winget >nul 2>&1
        if not errorlevel 1 (
            echo [INFO]   使用 winget 安装 ffmpeg ...
            winget install --id Gyan.FFmpeg --silent --accept-source-agreements --accept-package-agreements
        ) else (
            where choco >nul 2>&1
            if not errorlevel 1 (
                echo [INFO]   使用 choco 安装 ffmpeg ...
                choco install ffmpeg -y
            ) else (
                echo [WARN]   未检测到 winget 或 choco, 无法自动安装 ffmpeg
                echo          请手动下载: https://www.gyan.dev/ffmpeg/builds/
                echo          下载后将 bin 目录加入 PATH, 然后重新打开终端
            )
        )
        where ffmpeg >nul 2>&1
        if not errorlevel 1 (
            echo [ OK ] ffmpeg 安装成功 ^(若 PATH 未刷新, 请关闭当前终端并重新打开^)
        ) else (
            echo [WARN] ffmpeg 安装后仍未在 PATH, 视频上传功能将不可用
            echo        若刚刚已通过 winget/choco 安装成功, 请关闭当前终端并重新打开后再运行 start.bat
        )
    )
)

REM ---------- 探测推理设备 (cuda / mps / cpu) ----------
set "DETECT_PY=%TEMP%\qwen_asr_detect_device_%RANDOM%.py"
> "%DETECT_PY%" echo try:
>>"%DETECT_PY%" echo  ^ ^ ^ ^import torch
>>"%DETECT_PY%" echo  ^ ^ ^ ^if torch.cuda.is_available(): print("cuda")
>>"%DETECT_PY%" echo  ^ ^ ^ ^elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(): print("mps")
>>"%DETECT_PY%" echo  ^ ^ ^ ^else: print("cpu")
>>"%DETECT_PY%" echo except Exception:
>>"%DETECT_PY%" echo  ^ ^ ^ ^print("cpu")

set "DEVICE=cpu"
for /f "delims=" %%D in ('uv run --no-sync python "%DETECT_PY%" 2^>nul') do set "DEVICE=%%D"
del /q "%DETECT_PY%" >nul 2>&1

REM ---------- 根据设备/后端组装 kwargs ----------
set "AUTO_BACKEND_KWARGS="
set "AUTO_ALIGNER_KWARGS="
set "PASS_CUDA_DEVICES=1"

if /I "%BACKEND%"=="vllm" (
    if /I not "%DEVICE%"=="cuda" (
        echo [ERR ] vLLM 后端仅支持 NVIDIA CUDA GPU, 当前未检测到可用 CUDA 设备 ^(detected=%DEVICE%^)
        echo        请改用 --backend transformers, 或在带 NVIDIA GPU 的 Linux 主机上运行
        exit /b 1
    )
) else (
    if /I "%DEVICE%"=="cuda" (
        set AUTO_BACKEND_KWARGS={"device_map":"cuda:0","dtype":"bfloat16","max_inference_batch_size":4,"max_new_tokens":4096}
        set AUTO_ALIGNER_KWARGS={"device_map":"cuda:0","dtype":"bfloat16"}
    ) else if /I "%DEVICE%"=="mps" (
        set AUTO_BACKEND_KWARGS={"device_map":"mps","dtype":"float16","max_inference_batch_size":1,"max_new_tokens":4096}
        set AUTO_ALIGNER_KWARGS={"device_map":"mps","dtype":"float16"}
        set "PASS_CUDA_DEVICES=0"
    ) else (
        set AUTO_BACKEND_KWARGS={"device_map":"cpu","dtype":"float32","max_inference_batch_size":1,"max_new_tokens":4096}
        set AUTO_ALIGNER_KWARGS={"device_map":"cpu","dtype":"float32"}
        set "PASS_CUDA_DEVICES=0"
    )
)

REM 用户是否已经在 EXTRA_ARGS 里指定 --backend-kwargs / --aligner-kwargs
set "USER_HAS_BACKEND_KWARGS=0"
set "USER_HAS_ALIGNER_KWARGS=0"
echo %EXTRA_ARGS% | findstr /C:"--backend-kwargs" >nul && set "USER_HAS_BACKEND_KWARGS=1"
echo %EXTRA_ARGS% | findstr /C:"--aligner-kwargs" >nul && set "USER_HAS_ALIGNER_KWARGS=1"

REM ---------- 启动 WebUI ----------
echo.
echo [INFO] 启动 Qwen3-ASR Gradio WebUI ...
echo   ASR 模型     : %ASR_MODEL%
echo   后端         : %BACKEND%
echo   推理设备     : %DEVICE%
echo   监听地址     : %HOST%:%PORT%
if "%PASS_CUDA_DEVICES%"=="1" echo   CUDA 设备    : %CUDA_DEVICES%
echo   模型缓存     : %MODELS_DIR%
echo   HF_ENDPOINT  : %HF_ENDPOINT%
if "%WITH_ALIGNER%"=="1" echo   ForcedAligner: %ALIGNER_MODEL%
echo.

set "CUDA_FLAG="
if "%PASS_CUDA_DEVICES%"=="1" set "CUDA_FLAG=--cuda-visible-devices %CUDA_DEVICES%"

set "BACKEND_KW_FLAG="
if not "%AUTO_BACKEND_KWARGS%"=="" if "%USER_HAS_BACKEND_KWARGS%"=="0" set BACKEND_KW_FLAG=--backend-kwargs "%AUTO_BACKEND_KWARGS%"

set "ALIGNER_FLAG="
if "%WITH_ALIGNER%"=="1" (
    set "ALIGNER_FLAG=--aligner-checkpoint %ALIGNER_MODEL%"
    if not "%AUTO_ALIGNER_KWARGS%"=="" if "%USER_HAS_ALIGNER_KWARGS%"=="0" set ALIGNER_FLAG=!ALIGNER_FLAG! --aligner-kwargs "%AUTO_ALIGNER_KWARGS%"
)

uv run --no-sync qwen-asr-demo ^
    --asr-checkpoint %ASR_MODEL% ^
    --backend %BACKEND% ^
    %CUDA_FLAG% ^
    --ip %HOST% ^
    --port %PORT% ^
    %BACKEND_KW_FLAG% ^
    %ALIGNER_FLAG% %EXTRA_ARGS%

endlocal
