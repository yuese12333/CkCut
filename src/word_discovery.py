import os
import math
from collections import defaultdict
from tqdm import tqdm

class WordDiscoverer:
    def __init__(self, max_word_len: int = 4, min_freq: int = 5):
        """
        初始化新词发现器
        
        :param max_word_len: 候选词的最大长度（通常中文词汇长度在 2-4 之间）
        :param min_freq: 最低频次阈值。低于此频次的词块将被视为噪音过滤掉
        """
        self.max_word_len = max_word_len
        self.min_freq = min_freq
        
        # 使用 defaultdict(int) 来统计词频，比传统的字典或 Counter 在大规模循环中更快
        self.ngram_counts = defaultdict(int)
        
        # 记录语料库的总字数，这在后续计算概率 P(x) 时会用到
        self.total_chars = 0 

    def count_ngrams(self, corpus_path: str):
        """
        第一阶段：扫描语料库，利用滑动窗口提取并统计所有的 N-gram 频次
        """
        if not os.path.exists(corpus_path):
            print(f"❌ 错误: 找不到语料文件 {corpus_path}")
            return

        file_size = os.path.getsize(corpus_path)
        print(f"🔍 开始扫描语料并提取 {self.max_word_len}-gram...")

        with open(corpus_path, 'r', encoding='utf-8') as f:
            with tqdm(total=file_size, desc="统计词频", unit='B', unit_scale=True) as pbar:
                for line in f:
                    # 更新进度条 (按字节)
                    line_bytes = len(line.encode('utf-8'))
                    pbar.update(line_bytes)
                    
                    line = line.strip()
                    if not line:
                        continue
                        
                    length = len(line)
                    self.total_chars += length
                    
                    # 核心逻辑：滑动窗口提取 N-gram
                    for i in range(length):
                        # 从当前字 i 开始，往后截取长度为 1 到 max_word_len 的子串
                        for j in range(1, self.max_word_len + 1):
                            if i + j <= length:
                                word = line[i:i+j]
                                self.ngram_counts[word] += 1

        # --- 内存优化与噪音过滤 ---
        print(f"\n[初步统计] 共发现 {len(self.ngram_counts)} 种不同的 N-gram 组合。")
        print("🧹 正在清理低频噪音并释放内存...")
        
        # 过滤掉低于 min_freq 阈值的词，极大地减少后续计算 PMI 和 熵的计算量
        self.ngram_counts = {
            word: count 
            for word, count in self.ngram_counts.items() 
            if count >= self.min_freq
        }
        
        print(f"[清理完成] 过滤频次 < {self.min_freq} 的组合后，剩余 {len(self.ngram_counts)} 种候选词。")
        print(f"📊 语料库有效总字数: {self.total_chars}")

    def compute_pmi(self, min_pmi: float = 5.0):
        """
        第二阶段：计算候选词的内部凝固度 (PMI)，并过滤低凝固度词汇。
        
        :param min_pmi: PMI 阈值。值越大，要求字与字之间的绑定越紧密。（通常设在 3.0 到 5.0 之间）
        """
        print(f"\n🧠 开始计算候选词的内部凝固度 (PMI)...")
        
        # 用于存储通过 PMI 测试的词及其分数
        self.pmi_scores = {}
        
        # 遍历所有通过频次筛选的候选词
        for word, count in tqdm(self.ngram_counts.items(), desc="计算 PMI"):
            # 单字没有内部结构，不需要计算凝固度
            if len(word) < 2:
                continue
                
            # 计算整个词出现的概率 P(x,y)
            p_word = count / self.total_chars
            
            # 初始化该词的最小 PMI 为无穷大
            min_pmi_value = float('inf')
            
            # 寻找该词内部的“最弱连接点”
            for i in range(1, len(word)):
                left_part = word[:i]
                right_part = word[i:]
                
                # 获取左右两部分的频次
                left_count = self.ngram_counts.get(left_part, 0)
                right_count = self.ngram_counts.get(right_part, 0)
                
                # 理论上如果整体频次 >= 5，其子集频次必定 >= 5。
                # 加此判断是为了极致的安全，防止除零错误
                if left_count == 0 or right_count == 0:
                    continue
                    
                # 计算左右两部分独立出现的概率 P(x) 和 P(y)
                p_left = left_count / self.total_chars
                p_right = right_count / self.total_chars
                
                # 应用公式计算当前切分点的 PMI
                pmi = math.log2(p_word / (p_left * p_right))
                
                # 记录该词所有切分方式中的最小值
                min_pmi_value = min(min_pmi_value, pmi)
                
            # 如果最弱连接处的凝固度依然大于我们设定的阈值，就保留它！
            if min_pmi_value >= min_pmi:
                self.pmi_scores[word] = min_pmi_value
                
        print(f"\n🔪 基于 PMI >= {min_pmi} 过滤后，剩余 {len(self.pmi_scores)} 个高凝固度候选词。")

    def compute_entropy(self, corpus_path: str, min_entropy: float = 1.0):
        """
        第三阶段：计算候选词的左右边界信息熵，过滤掉自由度低的伪词，并最终生成词典。
        
        :param corpus_path: 清洗后的语料路径
        :param min_entropy: 信息熵阈值。通常设置在 0.5 ~ 2.0 之间。
        """
        print(f"\n🌳 开始收集候选词的左右邻居并计算边界熵...")
        
        # 记录每个词左边和右边出现过的字符及其频次
        left_neighbors = defaultdict(lambda: defaultdict(int))
        right_neighbors = defaultdict(lambda: defaultdict(int))
        
        # 提取当前所有合格的候选词集合，加速查找
        valid_candidates = set(self.pmi_scores.keys())
        
        file_size = os.path.getsize(corpus_path)
        with open(corpus_path, 'r', encoding='utf-8') as f:
            with tqdm(total=file_size, desc="收集邻居", unit='B', unit_scale=True) as pbar:
                for line in f:
                    pbar.update(len(line.encode('utf-8')))
                    line = line.strip()
                    length = len(line)
                    
                    # 再次使用滑动窗口，但这次只关注在我们名单里的词
                    for i in range(length):
                        for j in range(1, self.max_word_len + 1):
                            if i + j <= length:
                                word = line[i:i+j]
                                if word in valid_candidates:
                                    # 记录左邻居
                                    if i > 0:
                                        left_neighbors[word][line[i-1]] += 1
                                    # 记录右邻居
                                    if i + j < length:
                                        right_neighbors[word][line[i+j]] += 1

        # --- 计算熵并进行最终过滤 ---
        self.final_words = {}
        
        def calculate_entropy(neighbors_dict):
            """内部辅助函数：根据邻居的频次分布计算信息熵"""
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
            # 单字天然成词，我们通常直接保留，或者可以设置独立规则
            if len(word) == 1:
                self.final_words[word] = self.ngram_counts[word]
                continue
                
            l_entropy = calculate_entropy(left_neighbors[word])
            r_entropy = calculate_entropy(right_neighbors[word])
            
            # 一个真正的词，必须左边界和右边界的自由度都大于阈值
            if min(l_entropy, r_entropy) >= min_entropy:
                # 还可以综合 PMI 和 频次 给词汇打个综合分，这里我们简单保留频次
                self.final_words[word] = self.ngram_counts[word]

        print(f"\n🎉 历经九九八十一难！最终生成词汇量: {len(self.final_words)} 个。")

    def export_dict(self, output_path: str):
        """将机器自己学到的词典导出为 txt 文件"""
        # 按词频从高到低排序
        sorted_words = sorted(self.final_words.items(), key=lambda x: x[1], reverse=True)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for word, freq in sorted_words:
                # 格式参考结巴分词：词语 词频
                f.write(f"{word} {freq}\n")
        print(f"📁 专属词典已成功导出至: {output_path}")

# ================= 测试入口 =================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cleaned_corpus = os.path.join(BASE_DIR, "data", "processed", "merged_cleaned.txt")
    output_dict = os.path.join(BASE_DIR, "data", "output_dict", "my_dict.txt")
    
    # 1. 初始化
    discoverer = WordDiscoverer(max_word_len=4, min_freq=5)
    
    # 2. 提取 N-gram (频率)
    discoverer.count_ngrams(cleaned_corpus)
    
    # 3. 计算 PMI (凝固度)
    discoverer.compute_pmi(min_pmi=4.0)
    
    # 4. 计算 边界熵 (自由度)
    discoverer.compute_entropy(cleaned_corpus, min_entropy=1.2)
    
    # 5. 导出心血结晶
    discoverer.export_dict(output_dict)