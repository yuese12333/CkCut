import os
import argparse
from src.preprocess import preprocess_directory
from src.word_discovery import WordDiscoverer
from src.segmenter import AutoSegmenter
from src.evaluate import Evaluator

def main():
    parser = argparse.ArgumentParser(description="🚀 AutoSeg: 纯无监督中文新词发现与分词引擎")
    
    # 核心步骤控制
    parser.add_argument(
        '--step',
        type=str,
        choices=['preprocess', 'train', 'segment', 'evaluate', 'all'], 
        required=True,
        help="指定要执行的流水线步骤: preprocess, train, segment, evaluate(评测模型), all"
    )
    
    # 算法超参数配置 (赋予默认值，也可通过命令行随时覆盖)
    parser.add_argument('--max_len', type=int, default=4, help="候选词最大长度 (默认: 4)")
    parser.add_argument('--min_freq', type=int, default=5, help="最低词频阈值 (默认: 5)")
    parser.add_argument('--min_pmi', type=float, default=4.0, help="最低内部凝固度 PMI 阈值 (默认: 4.0)")
    parser.add_argument('--min_entropy', type=float, default=1.0, help="最低边界信息熵阈值 (默认: 1.0)")
    
    args = parser.parse_args()

    # ================= 1. 统一路径配置 =================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_CORPUS_DIR = os.path.join(BASE_DIR, "data", "raw_corpus")
    PROCESSED_FILE = os.path.join(BASE_DIR, "data", "processed", "merged_cleaned.txt")
    OUTPUT_DICT = os.path.join(BASE_DIR, "data", "output_dict", "my_dict_wiki.txt") # 准备迎接维基词典！

    # ================= 2. 步骤分发逻辑 =================

    # --- 步骤 A: 语料预处理 ---
    if args.step in ['preprocess', 'all']:
        print("\n" + "="*50)
        print("🛠️ 阶段 1/3: 启动数据清洗与合并流水线")
        print("="*50)
        preprocess_directory(RAW_CORPUS_DIR, PROCESSED_FILE)

    # --- 步骤 B: 训练生成词典 ---
    if args.step in ['train', 'all']:
        print("\n" + "="*50)
        print("🧠 阶段 2/3: 启动无监督新词发现引擎")
        print("="*50)
        if not os.path.exists(PROCESSED_FILE):
            print(f"❌ 错误: 找不到清洗后的语料 {PROCESSED_FILE}，请先执行 preprocess 步骤！")
            return
            
        discoverer = WordDiscoverer(max_word_len=args.max_len, min_freq=args.min_freq)
        discoverer.count_ngrams(PROCESSED_FILE)
        discoverer.compute_pmi(min_pmi=args.min_pmi)
        discoverer.compute_entropy(PROCESSED_FILE, min_entropy=args.min_entropy)
        discoverer.export_dict(OUTPUT_DICT)

    # --- 步骤 C: 交互式分词测试 ---
    if args.step in ['segment', 'all']:
        print("\n" + "="*50)
        print("🔪 阶段 3/3: 启动 DAG 工业级分词器 (交互模式)")
        print("="*50)
        if not os.path.exists(OUTPUT_DICT):
            print(f"❌ 错误: 找不到词典文件 {OUTPUT_DICT}，请先执行 train 步骤！")
            return
            
        segmenter = AutoSegmenter(OUTPUT_DICT)
        print("\n💡 分词引擎已就绪！请输入句子进行测试 (输入 'q' 或 'exit' 退出):")
        
        while True:
            try:
                text = input("\n👉 请输入: ").strip()
                if text.lower() in ['q', 'exit']:
                    print("👋 拜拜！测试结束。")
                    break
                if not text:
                    continue
                    
                words = segmenter.cut(text)
                print(f"✂️  分词结果: {' / '.join(words)}")
            except KeyboardInterrupt:
                print("\n👋 强制退出。")
                break

    # --- 步骤 D: 在标准测试集上进行评测 ---
    if args.step in ['evaluate', 'all']:
        print("\n" + "="*50)
        print("📈 阶段: 启动模型量化评测 (P/R/F1)")
        print("="*50)
        if not os.path.exists(OUTPUT_DICT):
            print(f"❌ 错误: 找不到词典文件 {OUTPUT_DICT}，请先执行 train 步骤！")
            return
            
        # 你需要去网上下载一个标准答案文件，比如 PKU 评测集，放到 data 目录下
        GOLD_TEST_FILE = os.path.join(BASE_DIR, "data", "test_gold.txt")
        
        if not os.path.exists(GOLD_TEST_FILE):
            print(f"⚠️ 找不到测试集 {GOLD_TEST_FILE}。")
            print("请准备一个用空格分好词的标准 txt 文件进行评测。")
        else:
            segmenter = AutoSegmenter(OUTPUT_DICT)
            Evaluator.run(segmenter, GOLD_TEST_FILE)

if __name__ == "__main__":
    main()