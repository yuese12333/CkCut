import os
import math

class AutoSegmenter:
    """基于词典和动态规划的中文分词器，支持最大概率路径切分。"""

    def __init__(self, dict_path: str):
        """
        初始化分词器。

        :param dict_path: 词典文件路径，每行格式：词 频次（空格分隔）
        """
        self.FREQ = {}          # 词频字典，同时作为前缀树（所有前缀均被加入，非词频次为0）
        self.total_freq = 0     # 总词频，用于计算词语的概率
        self.dict_path = dict_path
        self.initialize()

    def initialize(self):
        """加载词典，构建包含所有前缀的频次映射，以加速DAG构建。"""
        print(f"⏳ 正在加载词典并构建前缀树: {self.dict_path}")
        if not os.path.exists(self.dict_path):
            raise FileNotFoundError(f"找不到词典文件: {self.dict_path}")

        with open(self.dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ')
                if len(parts) >= 2:
                    word = parts[0]
                    freq = int(parts[1])
                    self.FREQ[word] = freq
                    self.total_freq += freq

                    # 将词的所有前缀也加入词典（频次为0），便于在构建DAG时快速判断前缀存在性
                    for i in range(1, len(word)):
                        prefix = word[:i]
                        if prefix not in self.FREQ:
                            self.FREQ[prefix] = 0

        print(f"✅ 词典加载完成！共计 {len(self.FREQ)} 个节点 (含前缀)，总有效词频: {self.total_freq}")

    def get_dag(self, sentence: str) -> dict:
        """
        为句子构建有向无环图（DAG），表示所有可能的词位。

        :param sentence: 待切分句子
        :return: DAG字典，键为起始位置，值为该位置可能词条的结束位置列表
        """
        dag = {}
        N = len(sentence)
        for start in range(N):
            candidates = []
            end = start
            frag = sentence[start]
            # 逐步扩展，检查前缀是否存在
            while end < N and frag in self.FREQ:
                if self.FREQ[frag] > 0:          # 只有频次>0才是真正的词
                    candidates.append(end)
                end += 1
                if end < N:
                    frag = sentence[start:end+1]
            # 若没有词典词，则视为单字词
            if not candidates:
                candidates.append(start)
            dag[start] = candidates
        return dag

    def calc_dp(self, sentence: str, dag: dict) -> dict:
        """
        动态规划 (Dynamic Programming) 求解最大概率路径
        """
        N = len(sentence)
        route = {}
        route[N] = (0.0, 0)
        
        log_total = math.log(self.total_freq) if self.total_freq > 0 else 1.0

        for idx in range(N - 1, -1, -1):
            candidates = []
            for end in dag[idx]:
                word = sentence[idx:end+1]
                
                # 【修复核心】安全获取词频
                freq = self.FREQ.get(word, 0)
                # 极其严格地防止 0 或负数：
                # 如果是纯前缀 (freq==0) 或完全没见过的生字 (get不到返回0)
                # 统一赋予极小频次 1，提供平滑概率，防止 log(0) 崩溃
                if freq <= 0:
                    freq = 1
                
                log_prob = math.log(freq) - log_total + route[end+1][0]
                candidates.append((log_prob, end))
            
            # 记录当前位置最优的 (最大概率值, 截断位置)
            route[idx] = max(candidates, key=lambda item: item[0])
            
        return route

    def cut(self, sentence: str) -> list:
        """
        执行分词，返回词序列。

        :param sentence: 输入句子
        :return: 分词结果列表
        """
        if not sentence:
            return []

        dag = self.get_dag(sentence)
        route = self.calc_dp(sentence, dag)

        words = []
        pos = 0
        N = len(sentence)
        while pos < N:
            next_pos = route[pos][1] + 1
            words.append(sentence[pos:next_pos])
            pos = next_pos
        return words


# ================= 测试入口 =================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dict_path = os.path.join(BASE_DIR, "data", "output_dict", "my_dict_primary.txt")

    segmenter = AutoSegmenter(dict_path)

    test_sentences = [
        "在这个从零开始的自然语言处理项目中，我们取得了巨大成功。",
        "这是一个测试，用来检验我们的分词效果到底好不好。",
        "今天天气真不错，我们一起去爬山吧。"
    ]

    print("\n🔪 开始初步分词测试:")
    print("=" * 50)
    for sent in test_sentences:
        words = segmenter.cut(sent)
        print(f"📝 原文: {sent}")
        print(f"✂️ 分词: {' / '.join(words)}")
        print("-" * 50)