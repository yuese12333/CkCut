# 🚀 CkCut : 混合驱动中文分词引擎

CkCut 是一个从底层手搓的中文分词与新词发现解决方案。它不依赖于 Jieba 等第三方分词库，通过结合统计学特征与随机过程模型，实现了从海量无标签文本中自动学习词汇并精准切分的功能。

# 🌟 核心特性

## 双引擎切分架构

**DAG + DP**：基于前缀树构建有向无环图，利用动态规划求解最大概率路径。

**HMM + Viterbi**：利用隐马尔可夫模型对连续单字进行序列标注（BMES），有效识别词典外的未登录词（OOV）。

**无监督新词发现**：通过点互信息（PMI）衡量词汇内聚度，通过信息熵（Entropy）衡量边界自由度。

## 工业级性能优化

**内存剪枝**：支持 GB 级语料处理，训练过程自动进行低频词剪枝与 GC 回收。

**多进程并行**：步骤 B 的 PMI 与边界熵计算使用多进程并行（`multiprocessing.Pool`），充分利用多核 CPU，显著缩短新词发现耗时。

**流式处理**：GUI 支持大文件「后台流式分词」，边读边切边存，内存开销极低。

**全功能可视化界面**：基于 PyQt6 开发，支持词典动态挂载、HMM 开关消融实验及大文件处理引导。

**自动化评测流水线**：一键计算精准率（P）、召回率（R）与 F1 值，支持多测试文件聚合评分。

# 🧠 算法原理

## 1. 新词发现 (Unsupervised Learning)

模型通过扫描原始语料，计算候选片段的凝固度与自由度：

· **内部凝固度 (PMI)**：

$$PMI(x, y) = \log_{2} \frac{P(x, y)}{P(x)P(y)}$$

· **边界信息熵 (Entropy)**：衡量词汇在左右语境中的不确定性。

## 2. 序列标注 (Supervised Learning)

HMM 引擎通过学习精标语料，统计状态转移概率 $A$ 与发射概率 $B$，利用 Viterbi 算法寻找最优状态序列：

$$S^{*} = \arg\max_{S} P(O|S)P(S)$$

# 🗂️ 目录结构

| 路径 | 说明 |
|------|------|
| README.md | 项目说明文档 |
| requirements.txt | 依赖包列表 (numpy, tqdm, regex, PyQt6 等) |
| main.py | 命令行流水线入口 (A/B/C/D/E 步骤) |
| main_visible.py | 带 GUI 的可视化分词器（导入/分词/导出） |
| main_train_visible.py | 训练流水线可视化（按钮执行 A–E、调参、日志） |
| CkCut_win.spec | Windows 目录版打包配置 (PyInstaller) |
| data/ | 数据目录 (除output_dict外均写入.gitignore) |
| ├─ raw_corpus/ | 原始语料 (TXT) |
| ├─ processed/ | 预处理合并结果 |
| ├─ HMM_train/ | HMM 训练语料（空格分词的 .txt） |
| ├─ evaluate/ | 评测用标准分词 .txt |
| └─ output_dict/ | 输出词典与 HMM 模型 |
| src/ | 核心源码 |
| ├─ preprocess.py | 语料清洗与合并 |
| ├─ word_discovery.py | 新词发现 (PMI + 熵，多进程并行) |
| ├─ segmenter.py | 分词引擎 (DAG + HMM) |
| ├─ hmm_trainer.py | HMM 有监督训练 |
| └─ evaluate.py | P/R/F1 评测 |

# 🚀 快速开始

## 环境安装

```bash
pip install -r requirements.txt
# 界面需要 PyQt6（若 requirements 中为 PyQt5 可改为 pip install PyQt6）
```

## 命令行流水线 (main.py)

步骤代号：**A** 预处理 | **B** 新词发现 | **C** HMM 训练 | **D** 评测 | **E** 交互分词

```bash
# 完整流程 A→B→C→D→E
python main.py --step all

# 仅 A→B→C→D（不进入交互 E，适合脚本/自动化）
python main.py --step ABCD

# 仅更新 HMM 并评测
python main.py --step CD

# 纯词典模式评测（禁用 HMM）
python main.py --step D --withouthmm

# 调参示例
python main.py --step B --max_len 4 --min_freq 5 --min_pmi 4.0 --min_entropy 1.0
```

## 神经网络 + CRF 分词流程 (main_nn_crf.py)

`main_nn_crf.py` 是独立于 `main.py` 的全新流程，使用 **BiLSTM-CRF** 做 BMES 序列标注分词，直接复用 `data/HMM_train` 与 `data/evaluate` 语料目录。

```bash
# 1) 训练（全量语料）
python main_nn_crf.py --mode train --epochs 10 --batch_size 16 --embedding_dim 128 --hidden_dim 256 --device cpu

# 2) 评测（输出 Precision / Recall / F1）
python main_nn_crf.py --mode eval --embedding_dim 128 --hidden_dim 256 --device cpu

# 3) 交互分词
python main_nn_crf.py --mode infer --embedding_dim 128 --hidden_dim 256 --device cpu
```

常用参数说明：

- `--mode`：`train` / `eval` / `infer`
- `--train_dir`：训练语料目录（默认 `data/HMM_train`）
- `--test_dir`：评测语料目录（默认 `data/evaluate`）
- `--output_dir`：模型与词表输出目录（默认 `data/output_nn_crf`）
- `--model_path`：模型权重路径（默认 `data/output_nn_crf/bilstm_crf.pt`）
- `--vocab_path`：字表路径（默认 `data/output_nn_crf/char_vocab.json`）
- `--max_samples`：训练样本上限，`0` 表示全量（适合快速冒烟调试）

快速冒烟示例（先小样本验证流程可跑通）：

```bash
python main_nn_crf.py --mode train --epochs 1 --batch_size 16 --embedding_dim 64 --hidden_dim 128 --max_samples 200 --device cpu
python main_nn_crf.py --mode eval --embedding_dim 64 --hidden_dim 128 --device cpu
```

## 可视化分词器 (main_visible.py)

```bash
python main_visible.py
```

- 支持智能导入 TXT、界面一键分词、导出结果；500KB 以上大文件自动走「后台流式分词」。
- 可勾选「启用 HMM 兜底」对比效果，支持动态挂载/导出词典。

## 训练流水线可视化 (main_train_visible.py)

```bash
python main_train_visible.py
```

- **超参数**：max_len、min_freq、min_pmi、min_entropy、是否禁用 HMM，界面中直接调整。
- **步骤按钮**：A 预处理、B 新词发现、C HMM 训练、D 评测、E 交互分词；另提供「A→B→C→D」一键顺序执行。
- **运行日志**：各步骤的标准输出实时显示在下方文本框（子进程调用 main.py，编码已兼容 Windows）。

# 📦 打包发布

- **Windows**：使用目录版打包，体积远小于单文件 exe。  
  `pyinstaller CkCut_win.spec` → 生成 `dist/CkCut/` 文件夹，内含 `CkCut.exe` 及依赖。
- **macOS**：在 Mac 上使用 PyInstaller 可生成 `.app` 包；需在 Mac 本机执行打包命令。

# 📜 许可证

本项目采用 MIT License 开源。
