import os
import math
import gc
import sys
from collections import defaultdict
from tqdm import tqdm

class WordDiscoverer:
    def __init__(self, max_word_len: int = 4, min_freq: int = 5):
        self.max_word_len = max_word_len
        self.min_freq = min_freq
        self.ngram_counts = defaultdict(int)
        self.total_chars = 0

    def count_ngrams(self, corpus_path: str):
        if not os.path.exists(corpus_path):
            print(f"❌ 错误: 找不到语料文件 {corpus_path}")
            return

        file_size = os.path.getsize(corpus_path)
        print(f"🔍 开始扫描语料并提取 {self.max_word_len}-gram...")

        line_count = 0
        # 核心超参数：每处理 30 万行执行一次内存清理
        prune_interval = 300000

        with open(corpus_path, 'r', encoding='utf-8') as f:
            with tqdm(total=file_size, desc="统计词频", unit='B', unit_scale=True) as pbar:
                for line in f:
                    line_bytes = len(line.encode('utf-8'))
                    pbar.update(line_bytes)
                    
                    line = line.strip()
                    if not line:
                        continue
                        
                    length = len(line)
                    self.total_chars += length
                    
                    # 核心逻辑：滑动窗口提取 N-gram
                    for i in range(length):
                        for j in range(1, self.max_word_len + 1):
                            if i + j <= length:
                                word = line[i:i+j]
                                self.ngram_counts[word] += 1
                                
                    line_count += 1
                    
                    # ================= 🚀 核心内存优化：定期剪枝 =================
                    if line_count % prune_interval == 0:
                        # 扫描过程中，先暂时删掉只出现过 1 次的垃圾组合，遏制内存爆炸
                        self._prune_dict(threshold=2)

        print(f"\n[初步统计] 语料扫描完毕。准备执行最终 {self.min_freq} 频次过滤...")
        # 所有文本扫描完毕后，执行最终你设定的严格阈值过滤
        self._prune_dict(threshold=self.min_freq)
        
        print(f"[清理完成] 过滤后剩余 {len(self.ngram_counts)} 种有效候选词。")
        print(f"📊 语料库有效总字数: {self.total_chars}")

    def _prune_dict(self, threshold: int):
        """
        内存优化辅助函数：重建字典，剔除低频垃圾词，并强制释放内存
        """
        new_counts = defaultdict(int)
        for k, v in self.ngram_counts.items():
            # ⚠️ 极度致命警告：单字(len==1)无论频次多低都绝对不能删！
            # 否则后续计算 PMI 提取左右切片时会发生 KeyError 或除以 0 崩溃。
            if len(k) == 1 or v >= threshold:
                new_counts[k] = v
        
        # 彻底切断旧字典的引用，并呼叫系统立刻回收这块内存
        del self.ngram_counts
        gc.collect()
        
        self.ngram_counts = new_counts

    def compute_pmi(self, min_pmi: float = 5.0):
        print(f"\n🧠 开始计算候选词的内部凝固度 (PMI)...")
        self.pmi_scores = {}
        
        for word, count in tqdm(self.ngram_counts.items(), desc="计算 PMI"):
            if len(word) < 2:
                continue
                
            p_word = count / self.total_chars
            min_pmi_value = float('inf')
            
            for i in range(1, len(word)):
                left_part = word[:i]
                right_part = word[i:]
                
                left_count = self.ngram_counts.get(left_part, 0)
                right_count = self.ngram_counts.get(right_part, 0)
                
                if left_count == 0 or right_count == 0:
                    continue
                    
                p_left = left_count / self.total_chars
                p_right = right_count / self.total_chars
                pmi = math.log2(p_word / (p_left * p_right))
                min_pmi_value = min(min_pmi_value, pmi)
                
            if min_pmi_value >= min_pmi:
                self.pmi_scores[word] = min_pmi_value
                
        print(f"\n🔪 基于 PMI >= {min_pmi} 过滤后，剩余 {len(self.pmi_scores)} 个高凝固度候选词。")
        # 步骤结束后，进行一次例行垃圾回收
        gc.collect()

    def compute_entropy(self, corpus_path: str, min_entropy: float = 1.0):
        print(f"\n🌳 开始收集候选词的左右邻居并计算边界熵...")
        
        left_neighbors = defaultdict(lambda: defaultdict(int))
        right_neighbors = defaultdict(lambda: defaultdict(int))
        valid_candidates = set(self.pmi_scores.keys())
        
        file_size = os.path.getsize(corpus_path)
        with open(corpus_path, 'r', encoding='utf-8') as f:
            with tqdm(total=file_size, desc="收集邻居", unit='B', unit_scale=True) as pbar:
                for line in f:
                    pbar.update(len(line.encode('utf-8')))
                    line = line.strip()
                    length = len(line)
                    
                    for i in range(length):
                        for j in range(1, self.max_word_len + 1):
                            if i + j <= length:
                                word = line[i:i+j]
                                if word in valid_candidates:
                                    if i > 0:
                                        left_neighbors[word][line[i-1]] += 1
                                    if i + j < length:
                                        right_neighbors[word][line[i+j]] += 1

        self.final_words = {}
        
        def calculate_entropy(neighbors_dict):
            total_count = sum(neighbors_dict.values())
            if total_count == 0:
                return 0.0
            entropy = 0.0
            for count in neighbors_dict.values():
                p = count / total_count
                entropy -= p * math.log2(p)
            return entropy

        print("🧮 正在计算熵值并生成最终词典...")
        for word in valid_candidates:
            if len(word) == 1:
                self.final_words[word] = self.ngram_counts[word]
                continue
                
            l_entropy = calculate_entropy(left_neighbors[word])
            r_entropy = calculate_entropy(right_neighbors[word])
            
            if min(l_entropy, r_entropy) >= min_entropy:
                self.final_words[word] = self.ngram_counts[word]

        print(f"\n🎉 最终生成词汇量: {len(self.final_words)} 个。")
        
        # ================= 🚀 收尾阶段内存清理 =================
        # 词典已生成，释放这三个巨大的邻居和分数缓存字典
        del left_neighbors
        del right_neighbors
        del self.pmi_scores
        gc.collect()

    def export_dict(self):
        """
        修改后的导出方法：交互式询问用户输入词典名称（不加后缀），自动添加.txt后缀并导出
        """
        # 交互式获取用户输入的词典名称
        while True:
            dict_name = input("\n请输入要导出的词典名称（无需加.txt后缀）：").strip()
            
            
            # 校验输入是否为空
            if not dict_name:
                print("❌ 错误：词典名称不能为空，请重新输入！")
                continue
            break
        
        # 自动添加 .txt 后缀
        output_path = f"{dict_name}.txt"
        
        # 排序并导出
        sorted_words = sorted(self.final_words.items(), key=lambda x: x[1], reverse=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for word, freq in sorted_words:
                f.write(f"{word} {freq}\n")
                
        print(f"📁 专属词典已成功导出至: {os.path.abspath(output_path)}")
