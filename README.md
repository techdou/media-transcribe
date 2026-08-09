# media-transcribe

通用本地音视频转录与时间戳工作流：把本地音视频文件批量转成高精度文字 + 字级时间戳（SRT/JSON）+ 带时间戳的逐句转录稿。

全程本地运行，数据不出机器，支持 30+ 语种。

## 何时使用

- 用户给本地音视频文件，要求转成文字/字幕/口播稿/逐句稿
- 需要字级或句级时间戳来定位内容
- 批量转录多个文件，或需要离线/高精度中文转录
- 访谈/会议/播客/讲座的文稿与纪要整理

### 何时不用（交给相邻 skill）

| 场景 | 使用 |
| --- | --- |
| ChatCut 项目内加字幕/开字幕/双语字幕 | `chatcut:transcription` |
| 已有转录，只要清理口误/叠字/重复 | `koubo-clean` |
| 只要口误识别与删除任务清单 | `AI剪口播` |
| 课程成片全流程（转录是其中一步） | `dou-ai-course-video` |

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

## 输出四件套

每个输入生成 4 个文件：

- `<名>.txt` —— 全文（按标点换行）
- `<名>.srt` —— 字幕（按句切分）
- `<名>.json` —— 字级时间戳 `[{text, start, end}]`（秒）
- `<名>_timed.md` —— 逐句转录稿（每行 `[MM:SS.mmm - MM:SS.mmm] 文本`）

## 常用参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--lang` | Chinese | 语种，支持 30+ |
| `--provider` | DML | ONNX 编码器后端：DML / CUDA / CPU / TRT |
| `--no-ts` | 关 | 关闭时间戳对齐（只要文本时提速） |
| `--chunk-size` | 40 | 分片秒数 |
| `--out-dir` | 输入同目录 | 输出目录，建议显式指定避免覆盖 |
| `--search` | - | 在逐句稿中定位短语并输出时间戳 |
| `--doctor` | - | 环境自检，不转录 |

## 目录结构

```
media-transcribe/
├── SKILL.md               # 技能主文档（Agent 入口）
├── README.md
└── scripts/
    └── oral_pipeline.py   # 转录主脚本
```

## 运行环境

- 引擎：[HaujetZhao/Qwen3-ASR-GGUF](https://github.com/HaujetZhao/Qwen3-ASR-GGUF)（Qwen3-ASR-1.7B 文本 + Qwen3-ForceAligner-0.6B 字级时间戳）
- 依赖：Python、onnxruntime（DML/CUDA/CPU）、ffmpeg
- 引擎/模型目录通过环境变量或 `.env` 配置（见仓库根目录 `.env.example`），仓库内不含本机路径

## 说明

- 转录输出默认与输入同目录，请用 `--out-dir` 隔离；输出目录不入库（已在 `.gitignore` 忽略）
- 首次使用：复制 `.env.example` 为 `.env` 并填写本机路径，或设置 `QWEN3_ASR_ENGINE_DIR` / `QWEN3_ASR_MODEL_DIR` 环境变量
