# 🚀 CkCut : 混合驱动中文分词引擎

CkCut 是一个从底层手搓的中文分词与新词发现解决方案。它不依赖于 Jieba 等第三方分词库，通过结合统计学特征与随机过程模型，实现了从海量无标签文本中自动学习词汇并精准切分的功能。

# 🌟 核心特性
## 双引擎切分架构：

**DAG + DP**：基于前缀树构建有向无环图，利用动态规划求解最大概率路径。

**HMM + Viterbi**：利用隐马尔可夫模型对连续单字进行序列标注（BMES），有效识别词典外的未登录词（OOV）。

**无监督新词发现**：通过点互信息（PMI）衡量词汇内聚度，通过信息熵（Entropy）衡量边界自由度。

## 工业级性能优化：

**内存剪枝**：支持 GB 级语料处理，训练过程自动进行低频词剪枝与 GC 回收。

**流式处理**：GUI 支持大文件“后台流式分词”，边读边切边存，内存开销极低。

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

# 🗂️ **目录结构规划**

## CkCut 项目目录结构

| 路径 | 说明 |
|------|------|
| README.md | 项目说明文档 |
| requirements.txt | 依赖包列表 (如 numpy, tqdm) |
| data/ | 数据存放目录 (需在 .gitignore 中忽略大文件) |
| ├─ evaluate/ | 评测数据集 |
| ├─ HMM_train/ | HMM训练数据集 |
| ├─ raw_corpus/ | 原始外部语料库 (TXT文件) |
| ├─ processed/ | 预处理后的清理数据 |
| └─ output_dict/ | 程序自行生成的词典输出目录 |
| src/ | 核心源代码目录 |
| ├─ preprocess.py | 语料清洗模块 (去除特殊字符、繁简转换等) |
| ├─ word_discovery.py | 核心：新词发现与词典生成模块 (统计 PMI 与 熵) |
| ├─ segmenter.py | 核心：基于生成词典的分词算法 (如最大匹配或 DAG+动态规划) |
| └─ evaluate.py | 评测模块 (计算 Precision, Recall, F1) |
| main.py | 项目的主入口脚本 |
|main_visible.py | 带GUI界面的可视化分词器 |

# 🚀 快速开始

环境安装
```Bash
pip install -r requirements.txt
```

命令行流水线 (CLI)

你可以通过字母代号自由组合执行步骤：

A: 预处理 | B: 新词发现 | C: HMM 训练 | D: 模型评测 | E: 交互测试

```Bash
# 执行完整流程 (清洗+造词+训练HMM+跑分+交互)
python main.py --step all

# 仅更新 HMM 模型并进行评测
python main.py --step CD

# 纯词典模式评测（禁用 HMM）
python main.py --step D --withouthmm
```

可视化客户端 (GUI)

```Bash
python main_visible.py
```

智能导入：500KB 以上文件自动触发“后台导出模式”。

动态控制：界面实时勾选“启用 HMM ”，即刻对比切分效果。

# 📜 许可证

本项目采用 MIT License 开源。