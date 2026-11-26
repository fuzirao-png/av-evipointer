# AV-EviPointer

带原声音轨的视频问答实验。项目把时间窗口、视觉对象和声学事件绑定成证据指针，用指针分配输入预算，再由一致性验证器检查答案引用的证据。

## 核心结果与状态

**公开基准待实测。** 下表是《MLLM 租卡与实验方案》第 4.3 节的实验前预测。它给出了组合方法的对照目标。

| 指标 | 朴素视听拼接 | 完整方法（指针 + 验证器） | 预测变化 |
| --- | ---: | ---: | ---: |
| 原始划分准确率 | 82–90% | 85–92% | +2–4 pp |
| MUSIC-AVQA 过滤硬子集准确率 | 70–80% | 75–86% | +3–8 pp |
| 证据忠实度 | 50–65% | 70–85% | +10–20 pp |
| 原始划分与硬子集准确率差 | 10–18 pp | 5–11 pp | 预计缩小 |
| 弃答率 | <2% | 5–15% | 预计上升 |

pp 表示百分点。方案独立给出了准确率与配对增量的预期，统计区间需等逐题数据后按视频簇计算。

## 问题与动机

视频中出现某个乐器，并不保证它正在发声。答对“哪把乐器先响”需要把可见对象与音轨中的起音时间对应起来。无声单帧与均匀多帧对照可以诊断视觉捷径，静音与经重新标注的音轨干预可以进一步检查模型对声音及其时间位置的依赖。数据划分参考 [MUSIC-AVQA](https://github.com/GeWu-Lab/MUSIC-AVQA) 和 [Music-AVQA-Hard](https://huggingface.co/datasets/gwkrsrch/music_avqa_hard)。

## 方法与核心创新点

1. **用原视频时间定位视听证据。** 每个指针包含窗口编号、起止时间、可见对象、声学事件和二者关系。规划器看低帧率概览，并分别排序视觉窗口和音频窗口。回答器读取两个排序选中的片段，返回答案和结构化证据。
2. **让指针决定预算花在哪段。** 按需选取时间窗口，在选中窗口提高视觉帧率；音频以相应时间窗裁切。默认共用一个累计输入 token 上限，计入规划、回答、验证和重试的完整序列，包括文本、时间标记及重复输入。每题输出实际用量、调用次数和超限状态。各组比较使用同一上限，各自的实际用量随策略变化。
3. **用引用证据验证答案。** 先检查指针时间范围、字段和可用模态，再让同一底座在单独调用中判断引用片段是否支持答案。完全无法验证时可重定位或弃答。验证器判定与标注评估独立，后者用时段 IoU 和证据字段匹配评分。

这里的实现按论文方案重建，使用冻结的 [WeMM-Video](https://github.com/Tencent/WeMM-Video) 和提示词规划。较低密度的音频编码、学习式定位器、训练版 Qwen-MusicAVQA 对照及音频前端替换仍待实现。

## 基线是怎么做的

所有模型组共用一个 BF16 底座、同一多模态处理器、同一答案格式和确定性解码。给定同一清单，各组处理完全相同的题目。

| 组 | 参数 | 输入和操作 |
| --- | --- | --- |
| 中央单帧 | `center-frame` | 取时间中点图像，交给同一模型，仅用视觉 |
| 均匀视觉 | `vision-only` | 均匀选择时间窗，输入这些窗口的视频帧 |
| 均匀音频 | `audio-only` | 均匀选择时间窗，输入原声音轨 WAV |
| 朴素视听拼接 | `concat-av` | 均匀选窗，先输入独立的视频片段，再输入独立的音频片段 |
| 原版视听 | `native-av` | 以单个带音轨视频输入底座，帧按处理器配置采样 |
| 均匀视听 | `uniform-av` | 均匀选窗，窗内由底座交错编码音频和画面。作为组合消融的共同起点 |

`native-av` 保留整条音轨，资源需求随时长变化。`uniform-av`、`pointer-only`、`verifier-only` 和 `full` 共用窗口数与累计输入上限，额外调用计入同一预算。`no-interleave` 使用独立视听序列，并保留用于定位的原视频时间元数据。

## 消融与组合实验

### 指针和验证器的 2×2 对照

`--demo` 使用 6 条可重复的符号样例检查流程。其中 4 条可作答，2 条需要弃答。下表为本地实际执行的冒烟测试，任务内容由固定规则构造。公开基准的消融结果待跑。

| 组合 | 参数 | 准确率（4 条可答） | 覆盖率（6 条总计） | 正确弃答（2 条不可答） | MUSIC-AVQA / Hard |
| --- | --- | ---: | ---: | ---: | --- |
| 均匀视听 | `uniform-av` | 2/4 | 6/6 | 0/2 | 待实测 |
| 仅指针 | `pointer-only` | 4/4 | 6/6 | 0/2 | 待实测 |
| 仅验证器 | `verifier-only` | 2/4 | 2/6 | 2/2 | 待实测 |
| 指针 + 验证器 | `full` | 4/4 | 4/6 | 2/2 | 待实测 |

在这组构造样例里，引入指针找回了均匀采样漏掉的窗口，验证器挡住了不可作答时的猜测。汇总见 [冒烟测试记录](results/synthetic-smoke.json)。正式实验可对相同题目比较 `full − pointer-only` 和 `verifier-only − uniform-av`，再计算两者之差以检验组合交互项。

### 单项消融和音轨干预

| 实验 | 代码 | 公开基准状态 |
| --- | --- | --- |
| 去音频指针 | `no-audio-pointer`，音频改为均匀选窗 | 待实测 |
| 去视觉指针 | `no-visual-pointer`，视觉改为均匀选窗 | 待实测 |
| 去原生视听交错 | `no-interleave`，编码时先视觉再音频 | 待实测 |
| 去一致性验证 | `pointer-only`，保留指针选窗 | 待实测 |
| 静音替换 | 同一视频设 `condition: "muted"`，保留音轨长度 | 待实测 |
| 音轨错位或替换 | 清单提供处理后的媒体，按实际可答性和答案重新标注 | 待实测 |

方案预计静音后，准确率下降幅度为仅视觉 `<2 pp`、全模态基座 `5–12 pp`、完整方法 `8–18 pp`。这些范围按原答案口径比较，若干预使原题不可回答，另报合理弃答。方案还预计上下文占用下降 `30–50%`，p95 首字延迟为均匀分配的 `1.3–2×`。当前脚本记录 p95 完整响应时间，首字延迟待另行测量。音频前端消融和 MUSIC-AVQA-R、Video-MME 音轨子集的模型结果待补。

方案第 4.5 节对音频前端的准确率变化预测以全局池化 PANNs 为参照。步进池化 Whisper 预计 `+5–15 pp`，窗口 Q-Former 预计 `+4–12 pp`，UniMambaMia 预计 `+2–6 pp`。这些前端涉及不同编码器和压缩方式，实测时需要记录实际音频 token 预算与输入时长。

## 快速开始

### CPU 冒烟测试

Python 3.10+，只用标准库。从仓库根目录运行。

```bash
python -m av_evipointer --demo --out runs/smoke.json
```

### 实际模型评估

使用 Python 3.12、带 BF16 支持的 NVIDIA GPU 和 FFmpeg（含 ffprobe 与 libx264）。依赖按 [WeMM-Video 的 Transformers 安装说明](https://github.com/Tencent/WeMM-Video#install-dependencies) 安装。先审核模型仓库的自定义代码，再用其 40 位提交哈希固定权重及处理器版本。以下示例使用 4B 先通流程，正式方案使用 9B。

```bash
MODEL=tencent/WeMM-Video-4B
REVISION=$(git ls-remote "https://huggingface.co/$MODEL" HEAD | cut -f1)
python -m av_evipointer --manifest runs/music-avqa.jsonl \
  --model "$MODEL" --revision "$REVISION" --limit 8 \
  --modes uniform-av pointer-only verifier-only full \
  --window-seconds 4 --slots 2 --input-budget 8192 --out runs/music-smoke.json
```

去掉 `--limit` 可运行清单全量测试。默认运行全部组。按开发集固定窗口时长、每个模态的窗口数和预算后，再开展正式测试。清单输出含逐题答案、证据、弃答原因、累计 token 和按划分/条件的汇总。标注证据以 `evidence` 字段提供后才计算证据指标。准确率以可答题为分母，弃答算错。无答案干预样本另报正确弃答率。

### 准备 MUSIC-AVQA 清单

从 [官方项目](https://gewu-lab.github.io/MUSIC-AVQA/) 获取视频，将 `<video_id>.mp4` 放在同一目录。使用 [固定版本测试标注](https://raw.githubusercontent.com/GeWu-Lab/MUSIC-AVQA/f72f0e8a9bb897826aa0dda934fdece339d4e62f/data/json/avqa-test.json)。如需过滤划分，从 [Hard 划分](https://huggingface.co/datasets/gwkrsrch/music_avqa_hard) 导出 `source_index` 列为整数 JSON 数组。`source_index` 对应官方文件的零基行位置，原始 `question_id` 可重复。

```bash
python tools/prepare_music_avqa.py --annotations data/avqa-test.json \
  --video-dir media/music-avqa --hard-indices data/hard-indices.json \
  --out runs/music-avqa.jsonl
```

不提供 `--hard-indices` 时仅生成原始划分。提供索引后，`original` 条件的总体汇总使用全部有媒体的原始题，`split=hard` 子组单列。准备脚本展开题目模板并写出缺失文件、音轨或已删除题目清单。标注和视频按原始项目条款获取。

也可提供自定义 JSONL。每行一题，路径可相对于清单，`duration` 和时间戳单位为秒。示例中的证据字段来自独立标注。

```json
{"id":"q1","video_id":"v1","video":"../media/v1.mp4","duration":8,"split":"probe","condition":"original","question":"Which instrument rings first?","answer":"violin","options":["violin","flute"],"evidence":[{"start":1.0,"end":1.8,"object":"violin","sound":"onset","relation":"first"}]}
```

不可答样本使用 `"answer": null`。可选 `options` 中的答案应填写选项文本。当前按规范化字符串匹配答案，并按时段 IoU 与对象、声音、关系标签匹配证据。结果可能随原论文的开放题判分器变化。

## 项目结构

```text
av_evipointer/
  __main__.py          清单读取、评估和逐题报告
  core.py              预算、证据解析、重试与消融策略
  backends.py          WeMM-Video 后端和 CPU 符号后端
tools/
  prepare_music_avqa.py   官方测试标注与本地视频对齐
results/
  synthetic-smoke.json    合成冒烟测试汇总
```
