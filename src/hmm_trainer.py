import os
import math
import json
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

class HMMTrainer:
    def __init__(self):
        self.states = ['B', 'M', 'E', 'S']
        
        # 使用字典记录频次
        self.start_counts = {s: 0 for s in self.states}
        self.trans_counts = {s1: {s2: 0 for s2 in self.states} for s1 in self.states}
        self.emit_counts = {s: defaultdict(int) for s in self.states}
        
        self.state_counts = {s: 0 for s in self.states} # 记录每个状态出现的总次数

    def _make_label(self, word):
        """将一个词转化为 BMES 标签序列"""
        if len(word) == 1:
            return ['S']
        else:
            return ['B'] + ['M'] * (len(word) - 2) + ['E']

    def train(self, corpus_dir: str):
        """
        批量读取目录下所有分好词的语料并全局统计频次
        :param corpus_dir: 存放空格分词 txt 文件的目录路径
        """
        corpus_path = Path(corpus_dir)
        if not corpus_path.exists() or not corpus_path.is_dir():
            print(f"❌ 找不到 HMM 训练语料目录: {corpus_dir}")
            return

        txt_files = list(corpus_path.rglob("*.txt"))
        if not txt_files:
            print(f"⚠️ 在 {corpus_dir} 下没有找到任何 .txt 文件。")
            return

        print(f"🧠 启动 HMM 模型批量训练 (共 {len(txt_files)} 个文件)...")
        
        line_count = 0
        # 使用基于文件数量的统一进度条
        with tqdm(txt_files, desc="HMM 训练总进度", unit="文件") as pbar:
            for file_path in pbar:
                pbar.set_postfix(file=file_path.name)
                
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                            
                        words = line.split()
                        chars = []
                        labels = []
                        
                        for word in words:
                            chars.extend(list(word))
                            labels.extend(self._make_label(word))
                        
                        # 确保字数和标签数一致，且不为空
                        if len(chars) != len(labels) or len(chars) == 0:
                            continue
                            
                        # 1. 统计初始状态
                        self.start_counts[labels[0]] += 1
                        
                        # 2. 统计转移和发射状态
                        for i in range(len(labels)):
                            current_state = labels[i]
                            current_char = chars[i]
                            
                            self.state_counts[current_state] += 1
                            self.emit_counts[current_state][current_char] += 1
                            
                            if i < len(labels) - 1:
                                next_state = labels[i+1]
                                self.trans_counts[current_state][next_state] += 1
                                
                        line_count += 1

        print(f"\n✅ 语料统计完成，共处理 {line_count} 行有效句子。")

    def save_model(self, output_json_path: str):
        """应用拉普拉斯平滑，计算对数概率，并保存模型"""
        print("🧮 正在计算对数概率并应用平滑机制...")
        
        start_p = {}
        trans_p = {}
        emit_p = {}
        
        MIN_FLOAT = -3.14e100
        
        # 1. 计算初始概率
        total_starts = sum(self.start_counts.values())
        for state in self.states:
            count = self.start_counts[state]
            if count == 0 and state in ['M', 'E']:
                start_p[state] = MIN_FLOAT
            else:
                prob = (count + 1) / (total_starts + 4)
                start_p[state] = math.log(prob)

        # 2. 计算转移概率
        for s1 in self.states:
            trans_p[s1] = {}
            total_trans = sum(self.trans_counts[s1].values())
            for s2 in self.states:
                count = self.trans_counts[s1][s2]
                if count == 0 and (
                    (s1 == 'B' and s2 in ['B', 'S']) or
                    (s1 == 'M' and s2 in ['B', 'S']) or
                    (s1 == 'E' and s2 in ['M', 'E']) or
                    (s1 == 'S' and s2 in ['M', 'E'])
                ):
                    trans_p[s1][s2] = MIN_FLOAT
                else:
                    prob = (count + 1) / (total_trans + 4)
                    trans_p[s1][s2] = math.log(prob)

        # 3. 计算发射概率
        for state in self.states:
            emit_p[state] = {}
            total_emit = self.state_counts[state]
            vocab_size = len(self.emit_counts[state])
            
            for char, count in self.emit_counts[state].items():
                prob = (count + 1) / (total_emit + vocab_size)
                emit_p[state][char] = math.log(prob)

        model_data = {
            'START_P': start_p,
            'TRANS_P': trans_p,
            'EMIT_P': emit_p
        }
        
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
            
        print(f"💾 专属 HMM 模型参数已成功导出至: {output_json_path}")

# ================= 测试入口 =================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 指向你的训练目录 HHM_train
    training_dir = os.path.join(BASE_DIR, "data", "HHM_train")
    output_model = os.path.join(BASE_DIR, "data", "output_dict", "hmm_model.json")
    
    trainer = HMMTrainer()
    trainer.train(training_dir)
    trainer.save_model(output_model)