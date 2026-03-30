# CkCut

CkCut 是一个中文分词实验项目，当前包含两条分词路线：

- **机械分词**：词典 + DAG + 动态规划 + 可选 HMM 兜底
- **深度学习分词**：BiLSTM-CRF（BMES 序列标注）

并提供一个 Flet 图形界面用于多模型对比展示。

---

## 目录结构

```text
CkCut/
├─ data/
│  ├─ output_dict/                 # 机械分词词典与 hmm_model.json
│  ├─ output_nn_crf_merged/        # 深度学习综合模型输出
│  ├─ output_nn_crf_single/        # 深度学习单语料模型输出（4 套）
│  ├─ HMM_train/                   # 训练语料（空格分词）
│  ├─ evaluate/                    # 评测语料
│  ├─ raw_corpus/                  # 原始语料
│  └─ processed/                   # 预处理产物
├─ src_machine/                    # 机械分词核心源码
├─ src_nn_crf/                     # BiLSTM-CRF 核心源码
├─ main_machine.py                 # 机械分词命令行入口
├─ main_nn_crf.py                  # 深度学习命令行入口
├─ main_flet.py                    # Flet GUI 入口（多模型对比）
└─ requirements.txt
```

---

## 环境安装

建议 Python 3.10+（Windows / Linux 均可）：

```bash
pip install -r requirements.txt
```

> 若使用 conda，请先激活目标环境再执行安装。

---

## 机械分词（main_machine.py）

`main_machine.py` 支持流水线步骤组合：

- A：预处理 raw_corpus
- B：无监督新词发现并生成词典
- C：训练 HMM
- D：评测（P/R/F1）
- E：交互式分词

### 示例

```bash
# 完整流程 A->B->C->D->E
python main_machine.py --step all

# 仅 A->B->C->D（不进入交互）
python main_machine.py --step ABCD

# 仅评测（纯词典，不启用 HMM）
python main_machine.py --step D --withouthmm

# 新词发现参数示例
python main_machine.py --step B --max_len 4 --min_freq 5 --min_pmi 4.0 --min_entropy 1.0
```

---

## 深度学习分词（main_nn_crf.py）

`main_nn_crf.py` 支持：

- `train`：训练
- `eval`：评测
- `infer`：交互式推理

### 示例

```bash
# 训练
python main_nn_crf.py --mode train --epochs 10 --batch_size 16 --embedding_dim 128 --hidden_dim 256 --device auto

# 评测
python main_nn_crf.py --mode eval --test_dir data/evaluate --embedding_dim 128 --hidden_dim 256 --device auto

# 交互分词
python main_nn_crf.py --mode infer --embedding_dim 128 --hidden_dim 256 --device auto
```

### 分语料分别训练（4 个模型）

```bash
python main_nn_crf.py --mode train --train_dir data/HMM_train --output_dir data/output_nn_crf_single --train_separately --epochs 10 --batch_size 16 --embedding_dim 128 --hidden_dim 256 --device auto
```

---

## 图形界面（main_flet.py）

Flet GUI 用于成果展示与多模型对比，支持：

- 机械 / 深度配置并排添加到队列
- 多模型同文本对比
- 导入 TXT
- 文本过大时“运行并导出到 TXT”
- 一键清空输入/输出
- 运行期间按钮锁定、防重复模型添加

### 启动

```bash
python main_flet.py
```

---

## 数据与模型约定

### 机械分词

- 词典：
  - `data/output_dict/my_dict_primary.txt`
  - `data/output_dict/my_dict_wiki.txt`
- HMM：
  - `data/output_dict/hmm_model.json`

### 深度学习

- 综合模型：
  - `data/output_nn_crf_merged/bilstm_crf.pt`
  - `data/output_nn_crf_merged/char_vocab.json`
  - `data/output_nn_crf_merged/train_history.json`
- 单模型：
  - `data/output_nn_crf_single/<as_train|cityu_train|msr_training|pku_training>/...`

---

## License

仅用于课程实验与学习交流。
