# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A gradio demo for Qwen3 ASR models.
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
import numpy as np
import torch
from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import SUPPORTED_LANGUAGES
from scipy.io.wavfile import read as wav_read


def _title_case_display(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("_", " ")
    return " ".join([w[:1].upper() + w[1:] if w else "" for w in s.split()])


def _build_choices_and_map(items: Optional[List[str]]) -> Tuple[List[str], Dict[str, str]]:
    if not items:
        return [], {}
    display = [_title_case_display(x) for x in items]
    mapping = {d: r for d, r in zip(display, items)}
    return display, mapping


def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {s}. Use bfloat16/float16/float32.")


def _normalize_audio(wav, eps=1e-12, clip=True):
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def _audio_to_tuple(audio: Any) -> Optional[Tuple[np.ndarray, int]]:
    """
    Accept gradio audio:
      - {"sampling_rate": int, "data": np.ndarray}
      - (sr, np.ndarray)  [some gradio versions]
    Return: (wav_float32_mono, sr)
    """
    if audio is None:
        return None

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    if isinstance(audio, tuple) and len(audio) == 2:
        a0, a1 = audio
        if isinstance(a0, int):
            sr = int(a0)
            wav = _normalize_audio(a1)
            return wav, sr
        if isinstance(a1, int):
            wav = _normalize_audio(a0)
            sr = int(a1)
            return wav, sr

    return None


# ---------------------------------------------------------------------------
# Video -> Audio extraction (via system ffmpeg)
# ---------------------------------------------------------------------------

def _extract_audio_from_video(video_path: str) -> Tuple[np.ndarray, int]:
    """
    Use system ``ffmpeg`` to extract a 16kHz mono PCM WAV from any video
    container/codec, then read it back into ``(np.ndarray float32 mono, sr)``.

    Raises ``gr.Error`` with a friendly Chinese message when ffmpeg is missing
    or when extraction fails, so that the gradio UI can show it nicely.
    """
    if not video_path or not os.path.exists(video_path):
        raise gr.Error("视频文件不存在或无法读取。")

    if shutil.which("ffmpeg") is None:
        raise gr.Error(
            "未检测到系统 ffmpeg, 无法从视频中提取音频。\n"
            "macOS:   brew install ffmpeg\n"
            "Ubuntu:  sudo apt-get install -y ffmpeg\n"
            "Windows: winget install --id Gyan.FFmpeg  (或 choco install ffmpeg -y)\n"
            "或重新运行 ./start.sh / start.bat 让安装脚本自动安装。"
        )

    out_path = os.path.join(
        tempfile.gettempdir(), f"qwen_asr_audio_{uuid.uuid4().hex}.wav"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        out_path,
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        raise gr.Error("ffmpeg 提取音频超时 (>10 分钟), 请尝试更短的视频。")

    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        raise gr.Error(f"ffmpeg 提取音频失败:\n{err or '(no stderr)'}")

    try:
        sr, data = wav_read(out_path)
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass

    wav = _normalize_audio(data)
    return wav, int(sr)


def _parse_audio_any(audio: Any) -> Union[str, Tuple[np.ndarray, int]]:
    """
    Convert various gradio inputs to ``(wav_float32_mono, sr)``:
      - ``gr.Audio(type="numpy")`` -> ``(sr:int, ndarray)``
      - ``gr.Video``                -> ``str`` filepath (handled by ffmpeg)
      - legacy ``{"sampling_rate","data"}`` dict
    """
    if audio is None:
        raise ValueError("Audio is required.")
    if isinstance(audio, str):
        return _extract_audio_from_video(audio)
    at = _audio_to_tuple(audio)
    if at is not None:
        return at
    raise ValueError("Unsupported audio input format.")


# ---------------------------------------------------------------------------
# SRT export utilities
# ---------------------------------------------------------------------------

# Punctuation that triggers segmentation. Whitespace is NOT counted (it merely
# separates tokens). Includes Chinese full-width and English half-width forms,
# plus 、 … —.
_SRT_PUNCT_CHARS = set("，。！？；：、…—,.!?;:")


def _format_srt_timestamp(seconds: float) -> str:
    """Format seconds (float) as an SRT timestamp ``HH:MM:SS,mmm``."""
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        seconds = 0.0
    total_ms = int(round(float(seconds) * 1000.0))
    hours, rem_ms = divmod(total_ms, 3600 * 1000)
    minutes, rem_ms = divmod(rem_ms, 60 * 1000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _build_srt_from_text_and_timestamps(
    result_text: str,
    timestamps: List[Dict[str, Any]],
) -> str:
    """
    Build an SRT subtitle string by splitting ``result_text`` on punctuation
    and aligning each segment with consecutive ``timestamps`` tokens.

    The timestamps list typically contains per-character (or per-word) tokens
    that exclude punctuation. We walk through ``result_text`` char-by-char and
    consume tokens whose ``text`` matches the current text position, flushing
    a subtitle whenever a punctuation character (or end of text) is reached.
    """
    if not result_text or not timestamps:
        return ""

    cleaned: List[Dict[str, Any]] = []
    for t in timestamps:
        if not isinstance(t, dict):
            continue
        txt = t.get("text", None)
        st = t.get("start_time", None)
        et = t.get("end_time", None)
        if txt is None or st is None or et is None:
            continue
        try:
            st_f = float(st)
            et_f = float(et)
        except (TypeError, ValueError):
            continue
        cleaned.append({"text": str(txt), "start_time": st_f, "end_time": et_f})

    if not cleaned:
        return ""

    segments: List[Tuple[str, float, float]] = []
    cur_chars: List[str] = []
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None

    def _flush() -> None:
        nonlocal cur_chars, cur_start, cur_end
        if cur_chars and cur_start is not None and cur_end is not None:
            # Preserve original punctuation; only strip surrounding whitespace.
            seg_text = "".join(cur_chars).strip()
            # Drop a segment that ended up with only punctuation/whitespace and
            # no real content (no timed token was ever appended).
            has_real = any(c not in _SRT_PUNCT_CHARS and not c.isspace() for c in seg_text)
            if seg_text and has_real:
                start = float(cur_start)
                end = float(cur_end)
                if end < start + 0.05:
                    end = start + 0.05
                segments.append((seg_text, start, end))
        cur_chars = []
        cur_start = None
        cur_end = None

    i_text = 0
    i_ts = 0
    n_text = len(result_text)
    n_ts = len(cleaned)

    while i_text < n_text:
        ch = result_text[i_text]

        if ch.isspace():
            i_text += 1
            continue

        if ch in _SRT_PUNCT_CHARS:
            # Attach the punctuation to the current segment (preserve original)
            # so the SRT reads naturally, then flush.
            if cur_chars and cur_start is not None:
                cur_chars.append(ch)
            i_text += 1
            _flush()
            continue

        if i_ts < n_ts:
            tok = cleaned[i_ts]
            tok_text = tok["text"]
            if tok_text and result_text.startswith(tok_text, i_text):
                if cur_start is None:
                    cur_start = tok["start_time"]
                cur_end = tok["end_time"]
                cur_chars.append(tok_text)
                i_text += len(tok_text)
                i_ts += 1
                continue
            # mismatch: text has an extra char OR ts has an extra token
            tok_first = tok_text[:1] if tok_text else ""
            if tok_first and result_text.find(tok_first, i_text) != -1:
                cur_chars.append(ch)
                i_text += 1
            else:
                i_ts += 1
            continue

        # text remains but no more timestamps -> still record character (no time)
        cur_chars.append(ch)
        i_text += 1

    _flush()

    if not segments:
        return ""

    # ------------------------------------------------------------------
    # Post-process timeline:
    #   1) Make sure segments are non-decreasing.
    #   2) Guarantee a minimum readable duration. Forced aligners sometimes
    #      collapse several tokens onto a single frame, producing 0--80ms
    #      segments that flash by unreadably. We:
    #        a) extend such a segment forward up to the start of the next
    #           segment (with a 50ms gap), and
    #        b) if there's still no room (the next segment also starts at the
    #           same anchor), merge this segment's text into the previous one
    #           rather than emitting a flicker-frame line.
    # ------------------------------------------------------------------
    MIN_SEG_DUR = 0.6  # absolute floor in seconds
    PER_CHAR_DUR = 0.18  # reading speed floor: 0.18s per character
    GAP = 0.05  # min gap between adjacent segments

    def _desired_dur(text: str) -> float:
        char_count = sum(1 for c in text if not c.isspace())
        return max(MIN_SEG_DUR, char_count * PER_CHAR_DUR)

    fixed: List[Tuple[str, float, float]] = []
    n = len(segments)
    i = 0
    while i < n:
        txt, st, et = segments[i]

        # Step 1: monotonic start
        if fixed and st < fixed[-1][1]:
            st = fixed[-1][1]

        # Step 2: try extending end to satisfy MIN_SEG_DUR / per-char floor.
        next_start = segments[i + 1][1] if i + 1 < n else None
        desired = _desired_dur(txt)

        if next_start is not None:
            upper = max(st, next_start - GAP)
        else:
            upper = st + desired  # last segment can extend freely

        target_end = max(et, st + desired)
        if next_start is not None:
            target_end = min(target_end, upper)

        # Step 3: if we still don't have enough room, merge into previous
        # segment (or, if no previous, into the next one).
        achievable = target_end - st
        if achievable < MIN_SEG_DUR * 0.5:  # < 0.3s, definitely a flicker
            if fixed:
                # Merge into previous: append this text and stretch end.
                ptxt, pst, pet = fixed[-1]
                merged_txt = ptxt + txt if ptxt.endswith((" ", "\n")) else ptxt + txt
                merged_end = max(pet, target_end)
                # Recompute desired for the merged text
                merged_desired = _desired_dur(merged_txt)
                if next_start is not None:
                    merged_end = min(
                        max(merged_end, pst + merged_desired),
                        max(pst + 0.05, next_start - GAP),
                    )
                else:
                    merged_end = max(merged_end, pst + merged_desired)
                fixed[-1] = (merged_txt, pst, merged_end)
                i += 1
                continue
            elif i + 1 < n:
                # Merge into next: prepend this text to next segment.
                ntxt, nst, net = segments[i + 1]
                segments[i + 1] = (txt + ntxt, st, net)
                i += 1
                continue
            # else: only segment overall and too short, fall through and emit it

        # Make sure end >= st + 50ms even if everything is degenerate
        if target_end < st + 0.05:
            target_end = st + 0.05

        fixed.append((txt, st, target_end))
        i += 1

    lines: List[str] = []
    for idx, (txt, st, et) in enumerate(fixed, start=1):
        lines.append(str(idx))
        lines.append(f"{_format_srt_timestamp(st)} --> {_format_srt_timestamp(et)}")
        lines.append(txt)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _safe_basename_for_srt(src_name: Optional[str]) -> str:
    if not src_name:
        return "transcript"
    name = Path(str(src_name)).name
    stem = Path(name).stem
    stem = "".join(c for c in stem if c.isprintable()).strip()
    return stem or "transcript"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-asr-demo",
        description=(
            "Launch a Gradio demo for Qwen3 ASR models (Transformers / vLLM).\n\n"
            "Examples:\n"
            "  qwen-asr-demo --asr-checkpoint Qwen/Qwen3-ASR-1.7B\n"
            "  qwen-asr-demo --asr-checkpoint Qwen/Qwen3-ASR-1.7B --aligner-checkpoint Qwen/Qwen3-ForcedAligner-0.6B\n"
            "  qwen-asr-demo --backend vllm --cuda-visible-devices 0\n"
            "  qwen-asr-demo --backend transformers --backend-kwargs '{\"device_map\":\"cuda:0\",\"dtype\":\"bfloat16\",\"attn_implementation\":\"flash_attention_2\"}'\n"
            "  qwen-asr-demo --backend vllm --backend-kwargs '{\"gpu_memory_utilization\":0.85}'\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True,
    )

    parser.add_argument("--asr-checkpoint", required=True, help="Qwen3-ASR model checkpoint path or HF repo id.")
    parser.add_argument(
        "--aligner-checkpoint",
        default=None,
        help="Qwen3-ForcedAligner checkpoint path or HF repo id (optional; enables timestamps when provided).",
    )

    parser.add_argument(
        "--backend",
        default="transformers",
        choices=["transformers", "vllm"],
        help="Backend for ASR model loading (default: transformers).",
    )

    parser.add_argument(
        "--cuda-visible-devices",
        default="0",
        help=(
            "Set CUDA_VISIBLE_DEVICES for the demo process (default: 0). "
            "Use e.g. '0' or '1'"
        ),
    )

    parser.add_argument(
        "--backend-kwargs",
        default=None,
        help=(
            "JSON dict for backend-specific kwargs excluding checkpoints.\n"
            "Examples:\n"
            "  transformers: '{\"device_map\":\"cuda:0\",\"dtype\":\"bfloat16\",\"attn_implementation\":\"flash_attention_2\",\"max_inference_batch_size\":32}'\n"
            "  vllm        : '{\"gpu_memory_utilization\":0.8,\"max_inference_batch_size\":32}'\n"
        ),
    )
    parser.add_argument(
        "--aligner-kwargs",
        default=None,
        help=(
            "JSON dict for forced aligner kwargs (only used when --aligner-checkpoint is set).\n"
            "Example: '{\"dtype\":\"bfloat16\",\"device_map\":\"cuda:0\"}'\n"
        ),
    )

    # Gradio server args
    parser.add_argument("--ip", default="0.0.0.0", help="Server bind IP for Gradio (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8000, help="Server port for Gradio (default: 8000).")
    parser.add_argument(
        "--share/--no-share",
        dest="share",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to create a public Gradio link (default: disabled).",
    )
    parser.add_argument("--concurrency", type=int, default=16, help="Gradio queue concurrency (default: 16).")

    # HTTPS args
    parser.add_argument("--ssl-certfile", default=None, help="Path to SSL certificate file for HTTPS (optional).")
    parser.add_argument("--ssl-keyfile", default=None, help="Path to SSL key file for HTTPS (optional).")
    parser.add_argument(
        "--ssl-verify/--no-ssl-verify",
        dest="ssl_verify",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to verify SSL certificate (default: enabled).",
    )

    return parser


def _parse_json_dict(s: Optional[str], *, name: str) -> Dict[str, Any]:
    if s is None or not str(s).strip():
        return {}
    try:
        obj = json.loads(s)
    except Exception as e:
        raise ValueError(f"Invalid JSON for {name}: {e}")
    if not isinstance(obj, dict):
        raise ValueError(f"{name} must be a JSON object (dict).")
    return obj


def _apply_cuda_visible_devices(cuda_visible_devices: str) -> None:
    v = (cuda_visible_devices or "").strip()
    if not v:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = v


def _default_backend_kwargs(backend: str) -> Dict[str, Any]:
    if backend == "transformers":
        return dict(
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=4,
            max_new_tokens=512,
        )
    else:
        return dict(
            gpu_memory_utilization=0.8,
            max_inference_batch_size=4,
            max_new_tokens=4096,
        )


def _default_aligner_kwargs() -> Dict[str, Any]:
    return dict(
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(override)
    return out


def _coerce_special_types(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if k == "dtype" and isinstance(v, str):
            out[k] = _dtype_from_str(v)
        else:
            out[k] = v
    return out


def build_demo(
    asr: Qwen3ASRModel,
    asr_ckpt: str,
    backend: str,
    aligner_ckpt: Optional[str] = None,
) -> Tuple[gr.Blocks, Any, str]:
    """Build the gradio demo UI."""
    # Static language list; doesn't require the ASR model to be loaded.
    lang_choices_disp, lang_map = _build_choices_and_map(list(SUPPORTED_LANGUAGES))
    lang_choices = ["Auto"] + lang_choices_disp

    has_aligner = bool(aligner_ckpt)

    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )
    css = ""

    with gr.Blocks() as demo:
        gr.Markdown(
            f"""
# 🎙️ Qwen3 ASR Demo

**Backend:** `{backend}` &nbsp;·&nbsp; **ASR:** `{asr_ckpt}` &nbsp;·&nbsp; **Aligner:** `{aligner_ckpt if aligner_ckpt else "(disabled)"}`
"""
        )

        with gr.Row():
            # =========================================================
            # LEFT: input panel
            # =========================================================
            with gr.Column(scale=2):
                gr.Markdown("### 📥 输入 / Input")
                with gr.Tabs():
                    with gr.Tab("🎵 Audio"):
                        audio_in = gr.Audio(
                            label="Audio Input (上传音频)",
                            type="numpy",
                        )
                    with gr.Tab("🎬 Video"):
                        video_in = gr.Video(
                            label="Video Input (上传视频, 自动抽取音频)",
                            sources=["upload"],
                            include_audio=True,
                        )
                        gr.Markdown(
                            "支持 mp4 / mov / mkv / webm / avi 等容器, "
                            "通过系统 ffmpeg 自动抽取 16kHz 单声道音频。"
                        )

                lang_in = gr.Dropdown(
                    label="Language (语种)",
                    choices=lang_choices,
                    value="Auto",
                    interactive=True,
                )
                if has_aligner:
                    ts_in = gr.Checkbox(
                        label="Return Timestamps (返回时间戳, 用于生成字幕)",
                        value=True,
                    )
                else:
                    ts_in = gr.State(False)

                btn = gr.Button(
                    "🎙️ Transcribe (开始识别)",
                    variant="primary",
                    size="lg",
                )

            # =========================================================
            # RIGHT: output panel
            # =========================================================
            with gr.Column(scale=3):
                gr.Markdown("### 📤 输出 / Output")
                with gr.Row():
                    out_lang = gr.Textbox(
                        label="Detected Language",
                        lines=1,
                        max_lines=1,
                        interactive=False,
                        scale=1,
                    )
                out_text = gr.Textbox(
                    label="Result Text (识别结果)",
                    lines=6,
                    max_lines=20,
                    placeholder="识别完成后此处会显示完整文本...",
                    interactive=False,
                )

                if has_aligner:
                    srt_btn = gr.DownloadButton(
                        label="💾 Export SRT (点击直接下载字幕文件)",
                        variant="primary",
                        size="lg",
                    )
                    srt_text_view = gr.Textbox(
                        label="SRT Subtitle (字幕原文)",
                        lines=14,
                        max_lines=40,
                        interactive=False,
                        placeholder=(
                            "识别完成后此处会自动显示完整 SRT 字幕文本 "
                            "(序号 + 时间 + 字幕)..."
                        ),
                    )
                    ts_view = gr.JSON(
                        label="Timestamps (时间戳, JSON)",
                    )
                else:
                    ts_view = gr.State(None)
                    srt_text_view = gr.State("")
                    srt_btn = gr.State(None)

        # Hidden state: name of most recently uploaded source file (for SRT filename)
        src_name_state = gr.State("")
        # Hidden state: full SRT text generated on every successful transcribe
        srt_text_state = gr.State("")

        def run(audio_upload: Any, video_upload: Any, lang_disp: str, return_ts: bool):
            # Audio Tab and Video Tab are mutually exclusive in the UI; whichever
            # the user filled in, we use it. If both are filled (rare), audio wins.
            if audio_upload is not None:
                audio_obj = _parse_audio_any(audio_upload)
                src_name = ""  # gr.Audio in numpy mode doesn't expose original filename
            elif video_upload:
                audio_obj = _parse_audio_any(video_upload)  # str path -> ffmpeg
                src_name = os.path.basename(str(video_upload))
            else:
                raise gr.Error("请先上传音频或视频。")

            language = None
            if lang_disp and lang_disp != "Auto":
                language = lang_map.get(lang_disp, lang_disp)

            return_ts = bool(return_ts) and has_aligner

            results = asr.transcribe(
                audio=audio_obj,
                language=language,
                return_time_stamps=return_ts,
            )
            if not isinstance(results, list) or len(results) != 1:
                raise RuntimeError(
                    f"Unexpected result size: {type(results)} "
                    f"len={len(results) if isinstance(results, list) else 'N/A'}"
                )

            r = results[0]
            language_out = getattr(r, "language", "") or ""
            text_out = getattr(r, "text", "") or ""

            if has_aligner:
                ts_payload: Optional[List[Dict[str, Any]]] = None
                srt_text = ""
                if return_ts:
                    ts_payload = [
                        dict(
                            text=getattr(t, "text", None),
                            start_time=getattr(t, "start_time", None),
                            end_time=getattr(t, "end_time", None),
                        )
                        for t in (getattr(r, "time_stamps", None) or [])
                    ]
                    # # Print timestamps to the server console for inspection.
                    # try:
                    #     print(
                    #         "[qwen-asr-demo] Timestamps:\n"
                    #         + json.dumps(ts_payload, ensure_ascii=False, indent=2),
                    #         flush=True,
                    #     )
                    # except Exception:
                    #     pass
                    if text_out and ts_payload:
                        try:
                            srt_text = _build_srt_from_text_and_timestamps(
                                text_out, ts_payload
                            )
                        except Exception:
                            srt_text = ""

                return (
                    language_out,
                    text_out,
                    ts_payload,
                    srt_text,
                    gr.update(value=None),  # reset DownloadButton: clear stale file
                    src_name,
                    srt_text,  # srt_text_state
                )
            else:
                return (
                    language_out,
                    text_out,
                    src_name,
                )

        def export_srt(srt_text: str, src_name: str):
            if not srt_text or not srt_text.strip():
                raise gr.Error(
                    "暂无字幕可导出。请先勾选 'Return Timestamps' 并执行识别。"
                )

            base = _safe_basename_for_srt(src_name)
            out_path = os.path.join(tempfile.gettempdir(), f"{base}.srt")
            with open(out_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(srt_text)
            return out_path

        if has_aligner:
            btn.click(
                run,
                inputs=[audio_in, video_in, lang_in, ts_in],
                outputs=[
                    out_lang,
                    out_text,
                    ts_view,
                    srt_text_view,
                    srt_btn,
                    src_name_state,
                    srt_text_state,
                ],
            )
            srt_btn.click(
                export_srt,
                inputs=[srt_text_state, src_name_state],
                outputs=[srt_btn],
            )
        else:
            btn.click(
                run,
                inputs=[audio_in, video_in, lang_in, ts_in],
                outputs=[out_lang, out_text, src_name_state],
            )

    return demo, theme, css


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    _apply_cuda_visible_devices(args.cuda_visible_devices)

    backend = args.backend
    asr_ckpt = args.asr_checkpoint
    aligner_ckpt = args.aligner_checkpoint

    user_backend_kwargs = _parse_json_dict(args.backend_kwargs, name="--backend-kwargs")
    user_aligner_kwargs = _parse_json_dict(args.aligner_kwargs, name="--aligner-kwargs")

    backend_kwargs = _merge_dicts(_default_backend_kwargs(backend), user_backend_kwargs)
    backend_kwargs = _coerce_special_types(backend_kwargs)

    forced_aligner: Optional[str] = None
    forced_aligner_kwargs: Optional[Dict[str, Any]] = None
    if aligner_ckpt:
        forced_aligner = aligner_ckpt
        aligner_kwargs = _merge_dicts(_default_aligner_kwargs(), user_aligner_kwargs)
        forced_aligner_kwargs = _coerce_special_types(aligner_kwargs)

    print(f"[qwen-asr-demo] Loading ASR model (backend={backend}) ...", flush=True)
    if backend == "transformers":
        asr = Qwen3ASRModel.from_pretrained(
            asr_ckpt,
            forced_aligner=forced_aligner,
            forced_aligner_kwargs=forced_aligner_kwargs,
            **backend_kwargs,
        )
    else:
        asr = Qwen3ASRModel.LLM(
            model=asr_ckpt,
            forced_aligner=forced_aligner,
            forced_aligner_kwargs=forced_aligner_kwargs,
            **backend_kwargs,
        )
    print("[qwen-asr-demo] ASR model loaded.", flush=True)

    demo, theme, css = build_demo(
        asr, asr_ckpt, backend, aligner_ckpt=aligner_ckpt
    )

    launch_kwargs: Dict[str, Any] = dict(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        ssl_verify=True if args.ssl_verify else False,
        theme=theme,
        css=css,
    )
    if args.ssl_certfile is not None:
        launch_kwargs["ssl_certfile"] = args.ssl_certfile
    if args.ssl_keyfile is not None:
        launch_kwargs["ssl_keyfile"] = args.ssl_keyfile

    demo.queue(default_concurrency_limit=int(args.concurrency)).launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
