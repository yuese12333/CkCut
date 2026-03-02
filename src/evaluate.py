import os
from tqdm import tqdm
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
    def run(segmenter: AutoSegmenter, gold_file_path: str):
        """
        运行评测并计算 P, R, F1
        
        :param segmenter: 已实例化的分词器对象
        :param gold_file_path: 标准分词语料文件路径（词与词之间用空格分隔）
        """
        if not os.path.exists(gold_file_path):
            print(f"❌ 错误: 找不到测试集文件 {gold_file_path}")
            return

        total_gold_words = 0      # 标准答案的总词数
        total_pred_words = 0      # 我们预测的总词数
        total_correct_words = 0   # 预测正确的总词数

        print(f"📊 开始在测试集上评测分词器...")
        
        file_size = os.path.getsize(gold_file_path)
        with open(gold_file_path, 'r', encoding='utf-8') as f:
            with tqdm(total=file_size, desc="评测进度", unit='B', unit_scale=True) as pbar:
                for line in f:
                    pbar.update(len(line.encode('utf-8')))
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
            
        # 计算三大指标
        precision = total_correct_words / total_pred_words
        recall = total_correct_words / total_gold_words
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print("\n" + "="*40)
        print("🏆 评测结果报告")
        print("="*40)
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
    
    # 假设你有一个带有标准分词的测试文件，词与词之间用空格隔开
    test_gold_path = os.path.join(BASE_DIR, "data", "raw_corpus", "test_gold.txt")
    
    segmenter = AutoSegmenter(dict_path)
    Evaluator.run(segmenter, test_gold_path)