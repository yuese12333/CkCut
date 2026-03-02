import os
import argparse
from src.preprocess import preprocess_directory
from src.word_discovery import WordDiscoverer
from src.segmenter import AutoSegmenter
from src.evaluate import Evaluator
from src.hmm_trainer import HMMTrainer

def main():
    parser = argparse.ArgumentParser(
        description="🚀 AutoSeg: 混合驱动 (DAG+HMM) 中文分词引擎",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--step',
        type=str,
        required=True,
        help="""指定要执行的流水线步骤组合 (可自由组合，如 'AB', 'BDE', 'all'):
A: 数据清洗与合并 (Preprocess raw_corpus)
B: 无监督新词发现与主词典生成 (Train Main Dict)
C: 有监督 HMM 模型批量训练 (Train HMM Model)
D: 模型量化评测 P/R/F1 (Evaluate test set)
E: 交互式分词终端 (Segment) - 注: 死循环，将自动置于最后执行
all: 一键顺次执行 A -> B -> C -> D -> E """
    )
    
    # 算法超参数配置
    parser.add_argument('--max_len', type=int, default=4, help="候选词最大长度 (默认: 4)")
    parser.add_argument('--min_freq', type=int, default=5, help="最低词频阈值 (默认: 5)")
    parser.add_argument('--min_pmi', type=float, default=4.0, help="最低内部凝固度 PMI 阈值 (默认: 4.0)")
    parser.add_argument('--min_entropy', type=float, default=1.0, help="最低边界信息熵阈值 (默认: 1.0)")
    
    # 【新增】可选控制开关
    parser.add_argument('--withouthmm', action='store_true', help="禁用 HMM 兜底模型 (仅使用纯 DAG 词典分词)")
    
    args = parser.parse_args()

    # ================= 1. 统一路径配置 =================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 预处理与无监督字典训练 (A, B)
    RAW_CORPUS_DIR = os.path.join(BASE_DIR, "data", "raw_corpus")
    PROCESSED_FILE = os.path.join(BASE_DIR, "data", "processed", "merged_cleaned.txt")
    OUTPUT_DICT = os.path.join(BASE_DIR, "data", "output_dict", "my_dict_wiki.txt")
    
    # HMM 模型训练 (C)
    HMM_TRAIN_DIR = os.path.join(BASE_DIR, "data", "HHM_train")
    HMM_MODEL = os.path.join(BASE_DIR, "data", "output_dict", "hmm_model.json")
    
    # 评测集 (D)
    TEST_DIR = os.path.join(BASE_DIR, "data", "evaluate")

    # ================= 2. 步骤指令解析 =================
    step_cmd = args.step.upper()
    
    run_a = 'A' in step_cmd or step_cmd == 'ALL'
    run_b = 'B' in step_cmd or step_cmd == 'ALL'
    run_c = 'C' in step_cmd or step_cmd == 'ALL'
    run_d = 'D' in step_cmd or step_cmd == 'ALL'
    run_e = 'E' in step_cmd or step_cmd == 'ALL'

    if not any([run_a, run_b, run_c, run_d, run_e]):
        print("❌ 错误: 无效的步骤组合。请包含 A, B, C, D, E 或输入 all。")
        return

    # ================= 3. 步骤分发逻辑 (强制顺序: A -> B -> C -> D -> E) =================

    # --- 步骤 A: 语料预处理 ---
    if run_a:
        print("\n" + "="*50)
        print("🛠️ 阶段 [A]: 启动数据清洗与合并流水线")
        print("="*50)
        preprocess_directory(RAW_CORPUS_DIR, PROCESSED_FILE)

    # --- 步骤 B: 训练生成主词典 ---
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
        # 不使用word_discovery的export_dict()，避免阻塞
        print(f"📁 正在自动导出词典至: {OUTPUT_DICT}")
        sorted_words = sorted(discoverer.final_words.items(), key=lambda x: x[1], reverse=True)
        
        dir_name = os.path.dirname(OUTPUT_DICT)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        with open(OUTPUT_DICT, 'w', encoding='utf-8') as f:
            for word, freq in sorted_words:
                f.write(f"{word} {freq}\n")
        print(f"✅ 词典已保存。")

    # --- 步骤 C: HMM 模型有监督训练 ---
    if run_c:
        print("\n" + "="*50)
        print("🤖 阶段 [C]: 启动 HMM 模型批量训练")
        print("="*50)
        if not os.path.exists(HMM_TRAIN_DIR) or not os.path.isdir(HMM_TRAIN_DIR):
            print(f"❌ 错误: 找不到 HMM 训练集目录 {HMM_TRAIN_DIR}。")
            print("请在该目录下放入用空格分好词的 .txt 文件。")
            return
            
        trainer = HMMTrainer()
        trainer.train(HMM_TRAIN_DIR)
        trainer.save_model(HMM_MODEL)

    # --- 步骤 D: 在标准测试集上进行评测 ---
    if run_d:
        print("\n" + "="*50)
        print("📈 阶段 [D]: 启动模型量化评测 (Precision / Recall / F1)")
        print("="*50)
        if not os.path.exists(OUTPUT_DICT):
            print(f"❌ 错误: 找不到词典文件 {OUTPUT_DICT}，请先执行 B 步骤！")
            return
            
        if not os.path.exists(TEST_DIR) or not os.path.isdir(TEST_DIR):
            print(f"⚠️ 找不到标准测试集目录 {TEST_DIR}。")
            print("请在该目录下放入用空格分好词的 .txt 测试文件。")
        else:
            # 根据是否带有 --withouthmm 参数决定是否加载 HMM 模型
            active_hmm_model = None if args.withouthmm else HMM_MODEL
            if args.withouthmm:
                print("⚠️ 注意: 已启用 --withouthmm，当前为纯 DAG 词典评测模式。")
                
            segmenter = AutoSegmenter(OUTPUT_DICT, active_hmm_model)
            Evaluator.run(segmenter, TEST_DIR)

    # --- 步骤 E: 交互式分词测试 ---
    if run_e:
        print("\n" + "="*50)
        print("🔪 阶段 [E]: 启动混合架构工业级分词器 (交互模式)")
        print("="*50)
        if not os.path.exists(OUTPUT_DICT):
            print(f"❌ 错误: 找不到词典文件 {OUTPUT_DICT}，请先执行 B 步骤！")
            return
            
        active_hmm_model = None if args.withouthmm else HMM_MODEL
        segmenter = AutoSegmenter(OUTPUT_DICT, active_hmm_model)
        
        mode_text = "纯 DAG" if args.withouthmm else "混合双引擎 (DAG+HMM)"
        print(f"\n💡 {mode_text} 已就绪！请输入句子进行测试 (输入 'q' 或 'exit' 退出):")
        
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