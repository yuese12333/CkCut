import os
from tqdm import tqdm
from pathlib import Path
from src.segmenter import AutoSegmenter

class Evaluator:
    @staticmethod
    def get_word_spans(words: list) -> set:
        """
        将分词列表转化为字符区间集合。
        例如 ["这", "是", "测试"] -> {(0, 1), (1, 2), (2, 4)}
        """
        spans = set()
        offset = 0
        for word in words:
            length = len(word)
            spans.add((offset, offset + length))
            offset += length
        return spans

    @staticmethod
    def run(segmenter: AutoSegmenter, test_dir: str):
        """
        运行批量评测并计算全局 P, R, F1
        
        :param segmenter: 已实例化的分词器对象
        :param test_dir: 存放标准分词 txt 文件的目录
        """
        test_path = Path(test_dir)
        if not test_path.exists() or not test_path.is_dir():
            print(f"❌ 错误: 找不到测试集目录 {test_dir}")
            return

        # 递归查找目录下所有的 .txt 文件
        txt_files = list(test_path.rglob("*.txt"))
        if not txt_files:
            print(f"⚠️ 警告: 在 {test_dir} 下没有找到任何 .txt 测试文件。")
            return

        total_gold_words = 0      # 标准答案的总词数
        total_pred_words = 0      # 我们预测的总词数
        total_correct_words = 0   # 预测正确的总词数

        print(f"📊 开始在测试集目录 '{test_dir}' (共 {len(txt_files)} 个文件) 上评测分词器...")
        
        # 使用基于文件数量的统一进度条
        with tqdm(txt_files, desc="全局评测进度", unit="文件") as pbar:
            for file_path in pbar:
                pbar.set_postfix(file=file_path.name)
                
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                            
                        # 1. 获取标准答案 (Gold)
                        gold_words = line.split()
                        if not gold_words:
                            continue
                            
                        # 2. 生成无空格的原始句子用于测试
                        raw_sentence = "".join(gold_words)
                        
                        # 3. 让我们的引擎进行预测
                        pred_words = segmenter.cut(raw_sentence)
                        
                        # 4. 转化为区间集合
                        gold_spans = Evaluator.get_word_spans(gold_words)
                        pred_spans = Evaluator.get_word_spans(pred_words)
                        
                        # 5. 统计正确数量（两集合的交集）
                        correct_spans = gold_spans.intersection(pred_spans)
                        
                        total_gold_words += len(gold_spans)
                        total_pred_words += len(pred_spans)
                        total_correct_words += len(correct_spans)

        # 防止除零错误
        if total_pred_words == 0 or total_gold_words == 0:
            print("⚠️ 评测集为空或切分结果为空！")
            return
            
        # 计算全局三大指标 (Micro-Average)
        precision = total_correct_words / total_pred_words
        recall = total_correct_words / total_gold_words
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print("\n" + "="*40)
        print("🏆 全局评测结果报告")
        print("="*40)
        print(f"参与评测文件数: {len(txt_files)} 个")
        print(f"标准答案总词数: {total_gold_words}")
        print(f"模型预测总词数: {total_pred_words}")
        print(f"命中正确总词数: {total_correct_words}")
        print("-" * 40)
        print(f"🎯 精确率 (Precision) : {precision * 100:.2f}%")
        print(f"🔍 召回率 (Recall)    : {recall * 100:.2f}%")
        print(f"⭐ F1-Score         : {f1_score * 100:.2f}%")
        print("="*40)

# ================= 简单测试入口 =================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dict_path = os.path.join(BASE_DIR, "data", "output_dict", "my_dict_wiki.txt")
    hmm_path = os.path.join(BASE_DIR, "data", "output_dict", "hmm_model.json")
    
    # 指向测试目录
    test_dir_path = os.path.join(BASE_DIR, "data", "evaluate")
    
    segmenter = AutoSegmenter(dict_path, hmm_path)
    Evaluator.run(segmenter, test_dir_path)