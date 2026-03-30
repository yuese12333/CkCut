# CkCut 中文分词系统 — 课程实验报告

**项目**：CkCut（混合驱动中文分词引擎）  
**说明**：本报告按 **机器分词**（词典 + 图模型统计方法）与 **神经网络分词**（BiLSTM-CRF）两条技术路线分别阐述；文末补充系统级评测、共性问题与参考文献。

---

# 第一部分：机器分词

本部分对应工程中的 **DAG + 动态规划**、**HMM + Viterbi（BMES）** 及词典构建、新词发现、HMM 训练等传统流水线（核心实现见 `src/segmenter.py`，命令行入口见 `main.py`）。

## 1.1 算法原理与改进

### 统计词典 + DAG + 动态规划

- **词典**：词条带频次；加载时为每个词登记**前缀**（前缀频次可为 0），便于在前缀词典上向前扩展候选词。
- **DAG**：对每个起始位置，在词典允许范围内列出可能的词尾，构成有向无环图。
- **DP**：从句末向句首做动态规划，在「词频 + 归一化」意义下求一条**对数概率最大**的切分路径。

### HMM + Viterbi（BMES）

- 对 DAG 结果中出现的**连续单字串**（词典难以合成多字词时），在开启 HMM 时使用 **BMES** 四状态与 **Viterbi** 解码，将单字序列标注并拼成词。
- **未登录字（OOV）**：发射概率对未见字做均匀平滑，使解码更多依赖转移概率。

### 无监督新词发现（`WordDiscoverer`，`src/word_discovery.py`）

流水线用于从**原始纯文本**（无分词标注）中自动筛出候选词，导出为「词 + 频次」词典，供后续 DAG 使用。实现上为 **N-gram 统计 → 频次剪枝 → 点互信息（PMI）→ 左右熵（边界自由度）** 四级结构，与 `main.py` 步骤 B 对应。

#### （1）候选生成与频次统计

- 对语料按行扫描，用**滑动窗口**提取长度 \(1 \sim L\) 的连续子串作为 N-gram 候选，其中最大词长 \(L\) 由 **`WordDiscoverer(max_word_len=...)`** 指定（**`main.py`** 中对应参数为 **`--max_len`**，默认 4）。
- 统计每个候选串在语料中出现的次数；同时累计 **`total_chars`**（语料有效总字数，用于将频次转为概率）。
- **内存控制**：扫描过程中按固定行数周期执行 **`_prune_dict`**，剔除低于阈值的低频组合；**单字**无论频次一律保留（否则后续 PMI 拆分时缺少子串频次会导致计算失败）。全量扫描结束后再用 **`min_freq`**（默认 5）做一次严格过滤。

#### （2）内部凝固度：点互信息 PMI

目的：过滤「只是两个高频片段偶然拼在一起」的字符串，保留**内部结合紧**的候选词。

对长度 \(\ge 2\) 的候选串 \(w\)，在其**每一种左右切分** \(w = a \,\|\, b\)（\(a,b\) 非空）上，用语料中的字级频次估计概率：

\[
P(w) = \frac{\mathrm{count}(w)}{\mathrm{total\_chars}},\quad
P(a) = \frac{\mathrm{count}(a)}{\mathrm{total\_chars}},\quad
P(b) = \frac{\mathrm{count}(b)}{\mathrm{total\_chars}}
\]

定义该切分下的 PMI（实现为以 2 为底的对数）：

\[
\mathrm{PMI}(a,b) = \log_2 \frac{P(w)}{P(a)\,P(b)}
\]

对 \(w\) 的**所有**合法切分计算 PMI，取**最小值** \(\min\_pmi\_value\)（「最弱一环」：任一切分若结合松散，则整体不视为凝固词）。若 \(\min\_pmi\_value \ge\) **`min_pmi`**，则保留该候选并记录得分（`WordDiscoverer.compute_pmi` 的形参默认 5.0；**`main.py --step B` 中 `--min_pmi` 默认 4.0**，以实际运行参数为准）。计算在 **`multiprocessing.Pool`** 中并行完成，主进程将 `ngram_counts` 与 `total_chars` 注入子进程只读使用。

> 与信息论标准 PMI\((x;y)=\log\frac{P(x,y)}{P(x)P(y)}\) 一致：此处 \(w\) 对应联合事件 \((a,b)\) 成串出现，用于衡量相对独立假设的超出量。

#### （3）边界自由度：左熵与右熵

目的：过滤**仍在任意延伸**的片段（如某长串的「前缀」常被同一字跟在后面），保留在左右语境上**用法已相对独立**的串。

对经 PMI 筛选后的候选 \(w\)，再次扫描语料：凡窗口内出现 \(w\) 且左侧有字，则统计 **\(w\) 左侧紧邻字符** 的频次分布；右侧同理。记左、右邻字分布的信息熵为 \(H_L(w)\)、\(H_R(w)\)（实现为 \(H = -\sum p\log_2 p\)）。

本实现采用 **\(\min(H_L, H_R) \ge \texttt{min\_entropy}\)**（默认 1.0）作为保留条件：左右两侧中**更差**的一侧也必须足够「开放」，避免边界过窄的假词。单字不依赖熵，直接按频次保留。

#### （4）导出

- 通过 **`export_dict`** 将 `final_words` 按频次降序写出为 `词 频次` 文本行，供 `AutoSegmenter` 加载。

#### 工程小结（新词发现）

- **多进程**：PMI 与熵两阶段均用进程池加速，并限制进程数（如不超过 8）以减轻 IO 争抢。
- **内存**：大规模 N-gram 表通过周期性剪枝 + `gc.collect()` 控制峰值；结束后释放邻居表与 PMI 中间结构。

### 改进与工程化（分词与词典）

- **`_tokenize_input`（`src/segmenter.py`）**：标点作硬边界，英文与数字连续块整块保留，再对**纯中文段**做 DAG+HMM，避免符号与拉丁字母干扰。
- 相比「纯 DAG 输出单字链」，HMM 对连续单字提供**序列级约束**，缓解部分未登录词与黏连。
- `AutoSegmenter` 支持**单文件**或**多词典顺序合并**（后者同词可后覆盖）；界面 `main_visible.py` 中可对 **Wiki 大词典 / 双小说训练词典** 二选一加载。

## 1.2 能处理的分词问题（歧义、未登录词）

| 类型 | 说明 |
|------|------|
| **歧义** | 在固定词典与频次下，DP 选出一条最大打分路径，可消解**部分**结构歧义；效果强依赖词典覆盖与频次是否合理。 |
| **未登录词** | 词典无合适长词时易出现连续单字或切分过粗/过细；**HMM** 对 DAG 后的**连续单字串**再切，缓解部分 OOV；字级未见有平滑。 |
| **英文、数字、标点** | 预处理阶段单独成段，不参与中文 DAG 内部切分，减少错误类型。 |

**小结**：**词典内歧义**主要靠 **DP 路径**；**OOV** 主要靠 **HMM 段**；整体仍受词典规模与 HMM 训练语料限制。

## 1.3 主要函数与模块

**分词引擎（`src/segmenter.py`）**

| 函数 / 类 | 作用 |
|-----------|------|
| `_tokenize_input` | 标点 / 英文数字 / 中文段切分。 |
| `viterbi` | HMM 解码（BMES）。 |
| `AutoSegmenter.initialize` | 读入词典，构建 `FREQ` 与前缀。 |
| `get_dag` / `calc_dp` | 构图与反向动态规划。 |
| `_cut_core` | DAG+DP 后对单字缓冲调用 HMM。 |
| `cut` | 对外主接口。 |

**新词发现（`src/word_discovery.py`）**

| 函数 / 类 | 作用 |
|-----------|------|
| `WordDiscoverer.count_ngrams` | 滑动窗口统计 N-gram 与总字数，周期性低频剪枝。 |
| `WordDiscoverer.compute_pmi` | 并行计算各候选 PMI，按 `min_pmi` 过滤。 |
| `WordDiscoverer.compute_entropy` | 收集左右邻字分布，按 `min_entropy` 过滤。 |
| `WordDiscoverer.export_dict` | 导出「词 频次」词典文件。 |
| `_pmi_worker` / `_entropy_worker` | 子进程内单候选计算任务。 |

**相关入口**：`main.py`（预处理、新词发现、HMM 训练、基于 `AutoSegmenter` 的评测步骤等）；`src/evaluate.py` 中 `Evaluator` 可对机器分词器做 **P/R/F1**（与神经网络共用同一评测逻辑）。

## 1.4 语料与资源

- **词典**：如 `data/output_dict/my_dict_wiki.txt`（大规模）、`my_dict_primary.txt`（可由两部小说语料训练得到）等，供 DAG 使用。  
- **HMM 训练语料**：`data/HMM_train/` 下空格分词 `.txt`，用于统计转移/发射概率。  
- **评测集**：`data/evaluate/`，每行空格分词；评测时拼成无空格串再切分，与 gold 做边界级对比。  
- **新词发现输入**：一般为 `main.py` 预处理后的合并文本（如 `data/processed/` 下），由 `WordDiscoverer` 读入**无标注纯文本**；超参数 **`--max_len` / `--min_freq` / `--min_pmi` / `--min_entropy`** 在 `main.py --step B` 中传入（见 `README` 示例）。

## 1.5 问题、优劣与改进（机器分词侧）

- **问题**：大词典加载耗时；DAG 强依赖**频次与覆盖**；HMM 特征简单，对长难 OOV、跨领域文本有限。  
- **新词发现**：N-gram 空间随 `max_word_len` 与语料规模**指数级膨胀**，需依赖剪枝与并行；PMI/熵阈值设置敏感，过严丢词、过松易引入噪声。  
- **优点**：可解释性强、无需 GPU、资源占用相对可控；新词发现可**无监督**扩充词典。  
- **改进**：领域词典与频次校准、用户词典优先、HMM 与更大规模标注训练；新词发现侧可尝试 **互信息变体、互信息+左右熵权重调参、或人工后处理**。

---

# 第二部分：神经网络分词（BiLSTM-CRF）

本部分对应 **`src_nn_crf/`** 与 **`main_nn_crf.py`**：以 **字向量 + 双向 LSTM + CRF** 做 **BMES 序列标注**，端到端学习切分边界。

## 2.1 算法原理与改进

- **字嵌入 + 双向 LSTM**：为每个字编码上下文特征，经线性层得到各标签发射分数；LSTM 输入侧使用 **`pack_padded_sequence` / `pad_packed_sequence`**，按真实句长打包，避免无效 padding 参与双向递推，减少无效计算。  
- **CRF 层**：学习标签转移约束；训练用前向算法（`logsumexp` 张量化）计算配分，解码用 **Viterbi**（见 `model.BiLSTMCRF`）。  
- **建模侧改进**：相对生成式 HMM 的发射独立性假设，判别式 BiLSTM-CRF 能利用更丰富的上下文；损失为整句负对数似然在 batch 上的平均（`neg_log_likelihood`）。

## 2.2 性能优化与工程改进

以下为 **`src_nn_crf/`** 与 **`main_nn_crf.py`** 中针对**训练速度、GPU 利用率、评测吞吐与推理延迟**所做的主要优化，与「算法正确性」正交，但对实际可用性影响显著。

### 训练阶段（`train.py`）

| 优化项 | 说明 |
|--------|------|
| **句长分桶批采样 `LengthBucketBatchSampler`** | 按句长将样本分桶、桶内按长度排序后再组 batch，使同一 batch 内长度接近，**降低 padding 比例**，减少无效矩阵运算与显存浪费。桶宽与 `batch_size * bucket_multiplier` 相关。 |
| **DataLoader** | 支持 **`num_workers` 多进程预取**、**`prefetch_factor`**、`persistent_workers=True`（`num_workers>0` 时），减轻主线程阻塞，提高 GPU 喂料效率。 |
| **`pin_memory` + `non_blocking=True`** | CUDA 上将张量拷入显存时使用异步拷贝，与默认同步拷贝相比有利于流水线重叠。 |
| **自动混合精度 AMP** | 在 CUDA 上启用 **`torch.autocast`**（默认 `bfloat16`），在多数算子上用低精度加速、降低显存；配合 **`clip_grad_norm`** 稳定训练。 |
| **`torch.compile`** | 在非 Windows 平台（Linux/WSL 等）对模型做 **`torch.compile`**，利用图优化进一步缩短单步耗时（Windows 上默认关闭以避免兼容风险）。 |
| **CUDA 算子与矩阵精度** | `torch.backends.cudnn.benchmark = True`、`torch.set_float32_matmul_precision("high")`，在固定输入尺寸场景下可选更快卷积/矩阵实现。 |
| **优化器与学习率** | **Adam** + **`ReduceLROnPlateau`**，按 epoch 损失自适应降 LR；**`optimizer.zero_grad(set_to_none=True)`** 减少显存碎片。 |
| **日志与可观测性** | **`log_interval`** 控制进度条上滑动平均 loss 的更新频率，平衡可读性与 `loss.item()` 同步开销。 |

### 推理与评测阶段（`infer.py` / `evaluate.py`）

| 优化项 | 说明 |
|--------|------|
| **批量推理 `cut_batch`** | 相对逐句 `cut`，将多条句子 padding 成矩阵一次前向，显著降低 **GPU kernel 启动与 Python 循环** 开销；评测脚本默认较大 **`batch_size`**（如 128）跑全测试集。 |
| **Viterbi 回溯** | 解码路径在 **单次同步** 下将 `lengths`、`backpointers` 等批量搬到 CPU/NumPy 再回溯，**避免在循环内反复 `.item()`** 导致 GPU-CPU 同步风暴。 |
| **`torch.inference_mode()`** | 推理关闭梯度与版本计数，较 `no_grad` 更省开销。 |

### 命令行与运行环境（`main_nn_crf.py`）

- **`--device auto|cpu|cuda`**：按需选用 GPU。  
- **`KMP_DUPLICATE_LIB_OK`、`OMP_NUM_THREADS`**：缓解 Windows 下多库 OpenMP 冲突与 CPU 过订阅（与科学计算栈常见实践一致）。  
- 其它可调：**`--use_amp`、`--pin_memory`、`--prefetch_factor`、`--clip_grad_norm`** 等，与 `TrainConfig` 字段一一对应。

### 小结

深度学习侧优化覆盖 **数据管线（分桶、多进程）— 计算图（AMP、compile）— 解码实现（批量 Viterbi、减少同步）— 评测（批量 cut）** 全链路；在保持 BiLSTM-CRF 结构不变的前提下，显著缩短**训练 epoch 时间与全量评测时间**，并提高 **GPU 利用率**。

## 2.3 能处理的分词问题（歧义、未登录词）

| 类型 | 说明 |
|------|------|
| **歧义** | 从数据中学习边界模式，对训练分布内常见歧义往往较强；受**训练域**与字表限制。 |
| **未登录词** | 对训练见过的字词组合更稳；**字表外字**需依赖 `UNK` 等策略（若实现），否则可能偏弱。 |
| **英文、数字、标点** | 若整句直接送 CRF，可能与机器分词分支的预处理**不一致**，属待统一改进点。 |

**小结**：强项在**数据分布内**的边界泛化；弱项在**域外、字表外**与**预处理不一致**带来的风格差异。

## 2.4 主要函数与模块

| 模块 | 作用 |
|------|------|
| `data_pipeline` | 语料读取、BMES、编码与批处理。 |
| `model.BiLSTMCRF` | 网络与 CRF 前向/解码。 |
| `infer.CRFSegmenter` | 加载 `*.pt` 与 `char_vocab.json`，提供 `cut` / **`cut_batch`**。 |
| `train` / `evaluate` | 训练与批量推理下的 P/R/F1 评测。 |

**入口**：`main_nn_crf.py`（`train` / `eval` / `infer`）；`main_visible.py` 中可通过下拉选择**合并训练或分语料**等预置权重；`main_nn_crf_train_eval_gui.py` 提供训练与评测图形界面（可选）。

## 2.5 语料、测试与产物

- **训练**：默认使用 `data/HMM_train/`（与机器分词 HMM 同源：空格分词转 BMES）；支持**按文件分别训练**输出多套 `bilstm_crf.pt`。  
- **评测**：默认 `data/evaluate/`，指标为 **Precision / Recall / F1**（与 `Evaluator` 一致）。  
- **产物**：`char_vocab.json`、`train_history.json`；**`.pt` 权重体积大**，通常不纳入 Git，本地训练生成（见仓库 `.gitignore`）。

## 2.6 问题、优劣与改进

- **问题**：`DataLoader` 与训练循环需一致，避免解包错误；`torch.load` 参数随 PyTorch 版本变化需兼容；依赖 CUDA 与驱动环境；**`torch.compile` 仅在部分平台启用**，跨环境需实测。  
- **优点**：效果潜力高，可吃到大规模监督数据红利；经 **§2.2** 所列优化后，训练与全量评测在 GPU 上可达到较好吞吐。  
- **改进**：与机器分词**统一预处理**；更大语料、预训练字向量或更强编码器；模型导出（ONNX）与量化便于部署；在 Windows 上可探索稳定后开启 compile 或改用 WSL2 训练。

---

# 系统级说明（两路线共用）

## 评测方式

- `src/evaluate.py`：将 gold 与 pred 转为**字符区间集合**，交集计为正确，汇总 **P、R、F1**。  
- **客观测试**：在 `data/evaluate` 上跑分；**定性例句**：可参考 `segmenter.py` 中 `__main__` 样例。

## 共同问题与对策

1. **Windows OpenMP / MKL**：多库链 OpenMP 可能冲突，可用 `KMP_DUPLICATE_LIB_OK`、限制 `OMP` 线程等缓解。  
2. **PyQt6 环境**：部分机器缺运行库导致 DLL 加载失败，属部署问题。  
3. **两路线预处理不一致**：建议在架构上对齐「标点 / 英文数字 / 中文」再分送不同引擎。

## 总体评价

| 维度 | 机器分词 | 神经网络分词 |
|------|----------|----------------|
| 可解释性 | 高（词典、路径、HMM 状态） | 相对较低（权重驱动） |
| 资源 | 主要 CPU + 词典内存 | GPU 友好，需字表与 `.pt` |
| 训练/评测吞吐 | 不依赖 PyTorch 栈 | 依赖 **分桶 batch、AMP、批量推理** 等优化（见 §2.2） |
| 适用 | 可控词典场景、轻量部署 | 有标注数据、追求效果上限 |

---

*文档版本：与仓库 `README.md` 及 `src/`、`src_nn_crf/` 实现对应；具体超参与路径以本地配置为准。*
