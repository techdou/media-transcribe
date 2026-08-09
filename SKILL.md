---
name: media-transcribe
description: 通用本地音视频转录与时间戳工作流。把本地音视频文件（口播课、录屏、访谈、会议、播客、讲座、公开课）批量转成高精度文字 + 字级时间戳（SRT/JSON）+ 带时间戳的逐句转录稿，纯本地离线，支持 30+ 语种。当任务需要本地/批量/离线/高精度转录、字幕生成、口播稿整理、按时间戳定位剪辑点时使用。ChatCut 项目内的字幕请用 chatcut:transcription；已有转录只做口播去口误用 koubo-clean；口误识别审查稿用 AI剪口播。触发词：转录、转写、ASR、字幕、时间戳、音视频转文字、本地转写、会议记录、访谈转写、播客转写、口播稿、Qwen3-ASR。
---

# 本地音视频转录（media-transcribe）

把任意音视频一次性转成标准四件套：全文文本、字级时间戳（SRT/JSON）、带时间戳的逐句转录稿。全程本地运行，数据不出机器。

## 何时使用 / 何时不用（Agent 匹配决策）

**使用**：
- 用户给本地音视频文件，要求转成文字/字幕/口播稿/逐句稿
- 需要字级或句级时间戳来定位内容（“这句话在第几秒”）
- 批量转录多个文件、或需要离线/高精度中文转录
- 访谈/会议/播客/讲座的文稿与纪要整理

**不用（交给相邻 skill）**：
- ChatCut 项目内加字幕/开字幕/双语字幕 → `chatcut:transcription`
- 已有转录，只要清理口误/叠字/重复 → `koubo-clean`
- 只要口误识别与删除任务清单 → `AI剪口播`
- 课程成片全流程（转录是其中一步） → `dou-ai-course-video`（内部如需本地预转录再回到本 skill）

## 快速开始

```powershell
# 0) 环境自检（Agent 调用前必跑；全 PASS 才继续）
python scripts\oral_pipeline.py --doctor

# 1) 转录：单文件 / 多文件 / 目录
python scripts\oral_pipeline.py a.mp4 b.mp3
python scripts\oral_pipeline.py --folder .\raw --out-dir .\transcripts

# 2) 验收
python scripts\oral_pipeline.py --search "关键词" --out-dir .\transcripts
```

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--lang` | Chinese | 语种，支持 English/Japanese/Korean/Cantonese 等 30+ |
| `--provider` | DML | ONNX 编码器后端：DML / CUDA / CPU / TRT |
| `--no-ts` | 关 | 关闭时间戳对齐（只要文本时提速） |
| `--n-ctx` | 2048 | LLM 上下文 |
| `--chunk-size` | 40 | 分片秒数（秒） |
| `--memory-num` | 1 | 记忆历史分片数 |
| `--out-dir` | 输入同目录 | 输出目录，建议显式指定避免覆盖 |
| `--search` | - | 在逐句稿中定位短语并输出时间戳 |
| `--doctor` | - | 环境自检，不转录 |

## 输出与验收标准

每个输入生成 4 个文件：
- `<名>.txt` —— 全文（按标点换行）
- `<名>.srt` —— 字幕（按句切分）
- `<名>.json` —— 字级时间戳 `[{text, start, end}]`（秒）
- `<名>_timed.md` —— 逐句转录稿（每行 `[MM:SS.mmm - MM:SS.mmm] 文本` + 纯文本区）

**验收**：4 个产物都存在且非空；`json` 时间戳项数 > 0（未 `--no-ts` 时）；`_timed.md` 每行格式正确；用 `--search` 抽查关键句能命中并给出时间戳。

## 下游处理（按需）

- 口播清理：按 `koubo-clean` 规范审校
- 课程成片：`dou-ai-course-video` 模板
- 字幕进 ChatCut：`edit_captions`
- 纪要整理：直接基于 `_timed.md` 的逐句稿

## 故障排查（分级）

| 现象 | 处理 |
|---|---|
| `--doctor` 有 FAIL | 按提示修复：模型文件、DLL、依赖、ffmpeg |
| 初始化访问冲突 / `llama_init_from_model` 报错 | DLL 与项目版本不匹配，用作者 v0.1 包配套 DLL 整套替换 `inference\bin` |
| 转录结果 0 字 | 先 `--doctor`；音频是否为空；确认未传错误 `--duration`（内部固定 None） |
| provider 初始化失败 | `--provider CPU --no-gpu` 兜底，或确认 `onnxruntime-directml` |
| 长音频出现重复循环 | `--temperature 0.5` 或增大 `--memory-num` |
| 专有名词听错 | 属 ASR 常态（MCP→MZP、WorkBuddy→Work Buddy 等），按项目术语表统一（口播课见 `koubo-clean` 品牌词表） |

## 运行环境（环境变量配置，仓库不含本机路径）

- 引擎源码：`$env:QWEN3_ASR_ENGINE_DIR`
- 模型目录：`$env:QWEN3_ASR_MODEL_DIR`
- llama.cpp DLL：`$env:QWEN3_ASR_ENGINE_DIR\qwen_asr_gguf\inference\bin`（作者 v0.1 包配套 Vulkan 版）
- 内核：[HaujetZhao/Qwen3-ASR-GGUF](https://github.com/HaujetZhao/Qwen3-ASR-GGUF)（Qwen3-ASR-1.7B 文本 + Qwen3-ForceAligner-0.6B 字级时间戳）

**配置方式**（任选其一，优先级：命令行参数 > 系统环境变量 > `.env`）：

1. 复制仓库根目录 `.env.example` 为 `.env`，填入本机路径（脚本启动时自动加载，不覆盖已有环境变量）
2. 或设置系统/用户级环境变量 `QWEN3_ASR_ENGINE_DIR`、`QWEN3_ASR_MODEL_DIR`
3. 或转录时传 `--engine-dir <路径> --model-dir <路径>`

## 常见坑

- **模型下载**：GitHub release 直链需 `--ssl-no-revoke --tlsv1.2`；后台下载易断，用 `$env:QWEN3_DOWNLOAD_WATCHER`（在 `.env` 配置）监视器自动续传。
- **chinese_itn**：PyPI 无真包，引擎根目录的恒等 shim 生效中（数字规整暂不启用）。
- **输出覆盖**：脚本直接覆盖同名输出，务必用 `--out-dir` 隔离。
