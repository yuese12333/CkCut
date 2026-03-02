📝 AutoSeg: 自发现中文分词工具 (Auto Chinese Segmenter)
📌 项目简介
本项目是一个完全从零开始的中文分词器。不依赖任何外部的分词词典（如jieba分词的 dict.txt），而是纯粹通过统计学方法，让程序通过阅读海量外部原始语料库，自己“学会”什么是词，并生成专属词典，最后基于该词典完成文本的分词任务。

🎯 核心特性
无词典依赖：基于无监督学习，从生语料中自动发现新词并构建词典。

统计学驱动：利用 N-gram 词频统计、点互信息（PMI）和左右信息熵（Entropy）进行词汇凝固度与自由度的计算。

轻量化全 Python 实现：架构清晰，适合在 VS Code 中进行二次开发和算法迭代。

🗂️ 目录结构规划
CkCut/
├── README.md               # 项目说明文档
├── requirements.txt        # 依赖包列表 (如 numpy, tqdm)
├── data/                   # 数据存放目录 (需在 .gitignore 中忽略大文件)
│   ├── raw_corpus/         # 原始外部语料库 (TXT文件)
│   ├── processed/          # 预处理后的清理数据
│   └── output_dict/        # 程序自行生成的词典输出目录
├── src/                    # 核心源代码目录
│   ├── __init__.py
│   ├── preprocess.py       # 语料清洗模块 (去除特殊字符、繁简转换等)
│   ├── word_discovery.py   # 核心：新词发现与词典生成模块 (统计 PMI 与 熵)
│   ├── segmenter.py        # 核心：基于生成词典的分词算法 (如最大匹配或 DAG+动态规划)
│   └── evaluate.py         # 评测模块 (计算 Precision, Recall, F1)
└── main.py                 # 项目的主入口脚本

