import os
import argparse
from src.preprocess import preprocess_directory
from src.word_discovery import WordDiscoverer
from src.segmenter import AutoSegmenter
from src.evaluate import Evaluator

def main():
    # 使用 RawTextHelpFormatter 让 help 信息支持换行显示，更美观
    parser = argparse.ArgumentParser(
        description="🚀 AutoSeg: 纯无监督中文新词发现与分词引擎",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 核心步骤控制：改为接收任意字符串组合
    parser.add_argument(
        '--step',
        type=str,
        required=True,
        help="""指定要执行的流水线步骤组合 (可自由组合，不区分大小写，如 'AB', 'bd', 'all'):
    A: 数据清洗与合并 (Preprocess)
    B: 无监督新词发现与词典生成 (Train)
    C: 交互式分词终端 (Segment) - 注: 死循环，将自动置于最后执行
    D: 模型量化评测 P/R/F1 (Evaluate)
    all: 一键执行 A -> B -> D -> C"""
    )
    
    # 算法超参数配置
    parser.add_argument('--max_len', type=int, default=4, help="候选词最大长度 (默认: 4)")
    parser.add_argument('--min_freq', type=int, default=5, help="最低词频阈值 (默认: 5)")
    parser.add_argument('--min_pmi', type=float, default=4.0, help="最低内部凝固度 PMI 阈值 (默认: 4.0)")
    parser.add_argument('--min_entropy', type=float, default=1.0, help="最低边界信息熵阈值 (默认: 1.0)")
    
    args = parser.parse_args()

    # ================= 1. 统一路径配置 =================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_CORPUS_DIR = os.path.join(BASE_DIR, "data", "raw_corpus")
    PROCESSED_FILE = os.path.join(BASE_DIR, "data", "processed", "merged_cleaned.txt")
    OUTPUT_DICT = os.path.join(BASE_DIR, "data", "output_dict", "my_dict_wiki.txt")
    GOLD_TEST_FILE = os.path.join(BASE_DIR, "data", "test_gold.txt")

    # ================= 2. 步骤指令解析 =================
    # 将输入统一转为大写，方便匹配
    step_cmd = args.step.upper()
    
    run_a = 'A' in step_cmd or step_cmd == 'ALL'
    run_b = 'B' in step_cmd or step_cmd == 'ALL'
    run_c = 'C' in step_cmd or step_cmd == 'ALL'
    run_d = 'D' in step_cmd or step_cmd == 'ALL'

    if not any([run_a, run_b, run_c, run_d]):
        print("❌ 错误: 无效的步骤组合。请包含 A, B, C, D 或输入 all。")
        return

    # ================= 3. 步骤分发逻辑 (顺序强制为: A -> B -> D -> C) =================

    # --- 步骤 A: 语料预处理 ---
    if run_a:
        print("\n" + "="*50)
        print("🛠️ 阶段 [A]: 启动数据清洗与合并流水线")
        print("="*50)
        preprocess_directory(RAW_CORPUS_DIR, PROCESSED_FILE)

    # --- 步骤 B: 训练生成词典 ---
    if run_b:
        print("\n" + "="*50)
        print("🧠 阶段 [B]: 启动无监督新词发现引擎")
        print("="*50)
        if not os.path.exists(PROCESSED_FILE):
            print(f"❌ 错误: 找不到清洗后的语料 {PROCESSED_FILE}，请先执行 A 步骤！")
            return
            
        discoverer = WordDiscoverer(max_word_len=args.max_len, min_freq=args.min_freq)
        discoverer.count_ngrams(PROCESSED_FILE)
        discoverer.compute_pmi(min_pmi=args.min_pmi)
        discoverer.compute_entropy(PROCESSED_FILE, min_entropy=args.min_entropy)
        discoverer.export_dict(OUTPUT_DICT)

    # --- 步骤 D: 在标准测试集上进行评测 ---
    if run_d:
        print("\n" + "="*50)
        print("📈 阶段 [D]: 启动模型量化评测 (Precision / Recall / F1)")
        print("="*50)
        if not os.path.exists(OUTPUT_DICT):
            print(f"❌ 错误: 找不到词典文件 {OUTPUT_DICT}，请先执行 B 步骤！")
            return
        
        if not os.path.exists(GOLD_TEST_FILE):
            print(f"⚠️ 找不到标准测试集 {GOLD_TEST_FILE}。")
            print("请准备一个用空格分好词的标准 txt 文件进行评测。")
        else:
            segmenter = AutoSegmenter(OUTPUT_DICT)
            Evaluator.run(segmenter, GOLD_TEST_FILE)

    # --- 步骤 C: 交互式分词测试 (置于末尾) ---
    if run_c:
        print("\n" + "="*50)
        print("🔪 阶段 [C]: 启动 DAG 工业级分词器 (交互模式)")
        print("="*50)
        if not os.path.exists(OUTPUT_DICT):
            print(f"❌ 错误: 找不到词典文件 {OUTPUT_DICT}，请先执行 B 步骤！")
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

if __name__ == "__main__":
    main()