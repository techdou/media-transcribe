#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""口播转录管线：音视频 -> Qwen3-ASR-GGUF -> txt/srt/json + 口播稿（句级时间戳）。

内核：HaujetZhao/Qwen3-ASR-GGUF（Qwen3-ASR 1.7B 出文本，Qwen3-ForceAligner 0.6B 出字级时间戳）。
用法：
  python oral_pipeline.py a.mp4 b.mp3
  python oral_pipeline.py --folder .\\raw
  python oral_pipeline.py --search "权限模式"
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path


MEDIA_EXTS = (".mp4", ".mov", ".mkv", ".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".wmv", ".aac")


def _load_dotenv() -> None:
    """从技能根目录或当前目录的 .env 加载配置（不覆盖已存在的环境变量）。"""
    candidates = [
        Path(__file__).resolve().parent.parent / ".env",
        Path.cwd() / ".env",
    ]
    for env_path in candidates:
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()

ENGINE_DIR = os.environ.get("QWEN3_ASR_ENGINE_DIR", "").strip()
MODEL_DIR = os.environ.get("QWEN3_ASR_MODEL_DIR", "").strip()


def doctor() -> int:
    """环境自检：模型文件 / DLL / 依赖 / provider / ffmpeg，全部 PASS 才建议转录。"""
    ok = True

    def check(name: str, cond: bool, hint: str = "") -> None:
        nonlocal ok
        if not cond:
            ok = False
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -> {hint}" if hint and not cond else ""))

    check("引擎目录", bool(ENGINE_DIR) and os.path.isdir(ENGINE_DIR),
          "未配置 QWEN3_ASR_ENGINE_DIR（.env 或环境变量）" if not ENGINE_DIR else ENGINE_DIR)
    check("模型目录", bool(MODEL_DIR) and os.path.isdir(MODEL_DIR),
          "未配置 QWEN3_ASR_MODEL_DIR（.env 或环境变量）" if not MODEL_DIR else MODEL_DIR)
    if os.path.isdir(MODEL_DIR):
        m = discover_model_files(MODEL_DIR)
        for k, v in m.items():
            check(f"模型文件 {k}", v is not None, "从 models release 下载并解压到模型目录")

    bin_dir = os.path.join(ENGINE_DIR, "qwen_asr_gguf", "inference", "bin")
    for dll in ("ggml.dll", "ggml-base.dll", "llama.dll"):
        check(f"DLL {dll}", os.path.exists(os.path.join(bin_dir, dll)),
              "从作者 v0.1 包拷贝配套 DLL 到 qwen_asr_gguf/inference/bin")
    check("DLL GPU 后端 (vulkan/cuda)",
          os.path.exists(os.path.join(bin_dir, "ggml-vulkan.dll")) or os.path.exists(os.path.join(bin_dir, "ggml-cuda.dll")),
          "需要 ggml-vulkan.dll 或 ggml-cuda.dll")
    check("chinese_itn shim", os.path.exists(os.path.join(ENGINE_DIR, "chinese_itn.py")),
          "引擎根目录需有 chinese_itn.py（PyPI 无真包时的恒等兜底）")

    for mod in ("onnxruntime", "gguf", "srt", "soundfile", "typer", "rich", "nagisa", "soynlp"):
        try:
            __import__(mod)
            check(f"依赖 {mod}", True)
        except ImportError:
            check(f"依赖 {mod}", False, f"pip install {mod}")

    try:
        import onnxruntime as ort
        provs = ort.get_available_providers()
        check("ONNX DML provider", "DmlExecutionProvider" in provs,
              "pip install onnxruntime-directml（或用 --provider CPU）")
    except Exception:
        check("ONNX DML provider", False, "onnxruntime 未安装")

    check("ffmpeg", shutil.which("ffmpeg") is not None, "安装 ffmpeg 并加入 PATH（mp4/m4a 等格式必需）")
    print("结论:", "全部就绪，可以转录" if ok else "存在 FAIL，先修复环境再转录")
    return 0 if ok else 1


def discover_model_files(model_dir: str) -> dict:
    """扫描模型目录，自动匹配 encoder/llm/aligner 文件（不依赖固定文件名）。"""
    files = os.listdir(model_dir)
    def pick(prefix: str, exts: tuple, prefer: tuple) -> str | None:
        cands = [f for f in files if f.startswith(prefix) and f.endswith(exts)]
        if not cands:
            return None
        for p in prefer:
            for c in cands:
                if p in c:
                    return c
        return sorted(cands)[0]

    asr_prefer = ("int4", "int8", "fp16", "fp32")
    llm_prefer = ("q5_k", "q4_k", "q8_0", "f16")
    return {
        "asr_frontend": pick("qwen3_asr_encoder_frontend", (".onnx",), asr_prefer),
        "asr_backend": pick("qwen3_asr_encoder_backend", (".onnx",), asr_prefer),
        "asr_llm": pick("qwen3_asr_llm", (".gguf",), llm_prefer),
        "align_frontend": pick("qwen3_aligner_encoder_frontend", (".onnx",), asr_prefer),
        "align_backend": pick("qwen3_aligner_encoder_backend", (".onnx",), asr_prefer),
        "align_llm": pick("qwen3_aligner_llm", (".gguf",), llm_prefer),
    }


def build_config(model_dir: str, provider: str, n_ctx: int, chunk_size: float,
                 memory_num: int, timestamp: bool):
    from qwen_asr_gguf.inference import ASREngineConfig, AlignerConfig
    m = discover_model_files(model_dir)
    missing = [k for k, v in m.items() if not v]
    if missing:
        raise FileNotFoundError(f"模型目录缺少文件: {missing}（目录: {model_dir}）")

    align_config = None
    if timestamp:
        align_config = AlignerConfig(
            model_dir=model_dir,
            encoder_frontend_fn=m["align_frontend"],
            encoder_backend_fn=m["align_backend"],
            llm_fn=m["align_llm"],
            onnx_provider=provider,
            llm_use_gpu=True,
            n_ctx=n_ctx,
        )
    return ASREngineConfig(
        model_dir=model_dir,
        encoder_frontend_fn=m["asr_frontend"],
        encoder_backend_fn=m["asr_backend"],
        llm_fn=m["asr_llm"],
        onnx_provider=provider,
        llm_use_gpu=True,
        n_ctx=n_ctx,
        chunk_size=chunk_size,
        memory_num=memory_num,
        verbose=False,
        enable_aligner=timestamp,
        align_config=align_config,
    )


def fmt_ts(t: float) -> str:
    t = max(0.0, t)
    ms = int(round(t * 1000))
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{m:02d}:{s:02d}.{ms:03d}"


def split_sentences(items: list) -> list:
    """把字级时间戳按句末标点聚合成句子（保留标点）。"""
    enders = re.compile(r"[。！？…；;!?]|\.\.\.$")
    sentences = []
    buf = []
    start = None
    for it in items:
        text = it["text"]
        if start is None:
            start = it["start"]
        buf.append(text)
        if enders.search(text):
            sentences.append({
                "text": "".join(buf).strip(),
                "start": start,
                "end": it["end"],
            })
            buf = []
            start = None
    if buf:
        sentences.append({
            "text": "".join(buf).strip(),
            "start": start,
            "end": items[-1]["end"],
        })
    return [s for s in sentences if s["text"]]


def write_timed_md(path: str, source_name: str, duration: float, sentences: list, plain_text: str) -> None:
    lines = [
        f"# 逐句转录稿：{source_name}",
        "",
        f"- 来源：{source_name}",
        f"- 时长：{duration:.1f} 秒",
        "- 引擎：Qwen3-ASR-1.7B + Qwen3-ForceAligner-0.6B（纯本地）",
        "",
        "## 逐句稿（带时间戳）",
        "",
    ]
    for s in sentences:
        lines.append(f"[{fmt_ts(s['start'])} - {fmt_ts(s['end'])}] {s['text']}")
    lines += ["", "## 纯文本", "", plain_text, ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def norm(s: str) -> str:
    return re.sub(r"[\s\W_]+", "", s.lower())


def search_phrase(phrase: str, out_dir: str) -> None:
    q = norm(phrase)
    hits = 0
    for pattern in ("*_timed.md", "*_oral.md"):
        for md in sorted(glob.glob(os.path.join(out_dir, pattern))):
            text = open(md, encoding="utf-8").read()
            for line in text.splitlines():
                m = re.match(r"^\[(\d\d:\d\d\.\d{3}) - (\d\d:\d\d\.\d{3})\] (.*)$", line)
                if m and q in norm(m.group(3)):
                    print(f"{Path(md).name}  {m.group(1)} - {m.group(2)}  {m.group(3)}")
                    hits += 1
    print(f"共 {hits} 处")


def collect_files(paths: list, folder: str | None) -> list:
    files = []
    if folder:
        for ext in MEDIA_EXTS:
            files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
            files.extend(glob.glob(os.path.join(folder, "**", f"*{ext}"), recursive=True))
    for p in paths:
        if os.path.isdir(p):
            for ext in MEDIA_EXTS:
                files.extend(glob.glob(os.path.join(p, f"*{ext}")))
        else:
            files.append(p)
    return sorted(set(os.path.abspath(f) for f in files))


def main() -> None:
    ap = argparse.ArgumentParser(description="口播转录管线（Qwen3-ASR-GGUF 内核）")
    ap.add_argument("files", nargs="*", help="音视频文件或目录")
    ap.add_argument("--folder", help="批量处理整个目录（递归）")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--engine-dir", default=ENGINE_DIR)
    ap.add_argument("--provider", default="DML", choices=["DML", "CUDA", "CPU", "TRT"])
    ap.add_argument("--lang", default="Chinese", help="如 Chinese / English，留空自动识别")
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--chunk-size", type=float, default=40.0)
    ap.add_argument("--memory-num", type=int, default=1)
    ap.add_argument("--no-ts", action="store_true", help="关闭时间戳对齐")
    ap.add_argument("--out-dir", help="输出目录（默认与输入同目录）")
    ap.add_argument("--search", help="在已有口播稿中定位短语并输出时间戳")
    ap.add_argument("--doctor", action="store_true", help="环境自检（模型/DLL/依赖/ffmpeg），不转录")
    args = ap.parse_args()

    if args.doctor:
        sys.exit(doctor())

    if args.search:
        search_phrase(args.search, args.out_dir or os.getcwd())
        return

    if not args.engine_dir or not args.model_dir:
        ap.error("未配置引擎/模型目录：请设置 QWEN3_ASR_ENGINE_DIR 和 QWEN3_ASR_MODEL_DIR"
                 "（技能目录 .env 或系统环境变量），或用 --engine-dir / --model-dir 指定")

    files = collect_files(args.files, args.folder)
    if not files:
        ap.error("未找到音视频文件")

    sys.path.insert(0, args.engine_dir)
    os.chdir(args.engine_dir)  # 保证 qwen_asr_gguf 包与 chinese_itn shim 可导入

    from qwen_asr_gguf.inference import QwenASREngine
    from qwen_asr_gguf.inference import exporters

    config = build_config(
        args.model_dir, args.provider, args.n_ctx, args.chunk_size,
        args.memory_num, timestamp=not args.no_ts,
    )
    print(f"模型目录: {args.model_dir} | Provider: {args.provider} | 时间戳: {'开' if not args.no_ts else '关'}")

    engine = QwenASREngine(config=config)
    t_start = time.time()
    try:
        for f in files:
            stem = Path(f).stem
            out_dir = args.out_dir or str(Path(f).parent)
            os.makedirs(out_dir, exist_ok=True)
            base_out = os.path.join(out_dir, stem)

            print(f"\n=== {Path(f).name} ===")
            t0 = time.time()
            res = engine.transcribe(
                audio_file=f,
                language=args.lang or None,
                context="",
                start_second=0.0,
                duration=None,
                temperature=0.4,
            )
            dt = time.time() - t0

            txt_out = f"{base_out}.txt"
            srt_out = f"{base_out}.srt"
            json_out = f"{base_out}.json"
            exporters.export_to_txt(txt_out, res)
            if res.alignment:
                exporters.export_to_srt(srt_out, res)
                exporters.export_to_json(json_out, res)

            items = [{"text": it.text, "start": it.start_time, "end": it.end_time} for it in (res.alignment or [])]
            sentences = split_sentences(items)
            plain = res.text.strip()
            audio_duration = items[-1]["end"] if items else 0.0
            oral_out = f"{base_out}_timed.md"
            write_timed_md(oral_out, Path(f).name, audio_duration, sentences, plain)
            print(f"耗时 {dt:.1f}s | 文本 {len(plain)} 字 | 时间戳 {len(items)} 项 | 句子 {len(sentences)} 句")
            print(f"输出: {txt_out} / {srt_out} / {json_out} / {oral_out}")
    finally:
        engine.shutdown()
    print(f"\n全部完成，总耗时 {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
