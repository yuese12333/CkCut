import os
import math
import json
from typing import Optional, List, Tuple

# ================= 预处理：英文/数字/标点 与 纯中文 分离 =================
# 标点（全角+半角）：作为硬截断，单独成 token
# 特别包含中文全角引号：‘ ’
PUNCT_CHARS = set(
    "，。！？、；：\"\"''（）【】《》…—‘’\t\n "
    + ",.!?;:\"'()[]"
)


def _is_en_num_char(c: str) -> bool:
    """是否为英文、数字或 C++ 风格中的 +"""
    return c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+"


def _tokenize_input(s: str) -> List[Tuple[str, str]]:
    """
    将输入切分为 (类型, 内容) 列表。
    类型: "en_num" | "punct" | "chinese"
    - en_num: 连续英文/数字/加号（如 PyTorch, C++, 2026）整块保留
    - punct: 标点符号，每个单独成段
    - chinese: 仅含中文等非英文数字非标点的片段，送核心分词
    """
    if not s:
        return []
    segments = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] in PUNCT_CHARS:
            segments.append(("punct", s[i]))
            i += 1
        elif _is_en_num_char(s[i]):
            j = i
            while j < n and _is_en_num_char(s[j]):
                j += 1
            segments.append(("en_num", s[i:j]))
            i = j
        else:
            j = i
            while j < n and s[j] not in PUNCT_CHARS and not _is_en_num_char(s[j]):
                j += 1
            if j > i:
                segments.append(("chinese", s[i:j]))
            i = j
    return segments


# ================= HMM 核心解码组件 =================
# HMM 的四大状态：B(开头), M(中间), E(结尾), S(单字)
STATES = ['B', 'M', 'E', 'S']
MIN_FLOAT = -3.14e100

def viterbi(obs, states, start_p, trans_p, emit_p):
    """
    Viterbi 算法求解最大概率状态序列
    :param obs: 观察序列 (连续的单字字符串)
    """
    V = [{}] # 动态规划表格
    path = {} # 记录路径
    
    # 应对 OOV 字的平滑惩罚机制
    def get_emit_prob(state, char):
        if state in emit_p and char in emit_p[state]:
            return emit_p[state][char]
        # 如果字完全没见过，分配均等极小概率，逼迫算法依赖转移概率
        return math.log(1.0 / len(states))

    # 初始化 (t=0)
    for y in states:
        V[0][y] = start_p.get(y, MIN_FLOAT) + get_emit_prob(y, obs[0])
        path[y] = [y]

    # 递推 (t > 0)
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
        for y in states:
            em_p = get_emit_prob(y, obs[t])
            candidates = []
            for y0 in states:
                if y in trans_p.get(y0, {}):
                    prob = V[t-1][y0] + trans_p[y0][y] + em_p
                    candidates.append((prob, y0))
                else:
                    candidates.append((MIN_FLOAT, y0))
                    
            prob, state = max(candidates, key=lambda x: x[0])
            V[t][y] = prob
            # 安全获取路径
            newpath[y] = path.get(state, []) + [y]
            
        path = newpath

    # 终止 (最后一个字的状态只能是 E 结尾 或 S 单字)
    prob, state = max([(V[len(obs) - 1][y], y) for y in ('E', 'S')])
    return prob, path[state]


# ================= 工业级分词引擎 =================
class AutoSegmenter:
    """基于 DAG(词典查表) 与 HMM(隐马尔可夫模型) 双引擎的中文分词器"""

    def __init__(self, dict_path: str, hmm_model_path: Optional[str] = None):
        self.FREQ = {}
        self.total_freq = 0
        self.dict_path = dict_path
        
        # HMM 参数容器
        self.start_p = {}
        self.trans_p = {}
        self.emit_p = {}
        self.hmm_enabled = False
        
        self.initialize()
        
        # 如果传入了 hmm 模型路径，则加载
        if hmm_model_path and os.path.exists(hmm_model_path):
            self.load_hmm(hmm_model_path)

    def initialize(self):
        """加载词典，构建包含所有前缀的频次映射"""
        print(f"⏳ 正在加载主词典并构建前缀树: {self.dict_path}")
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

                    for i in range(1, len(word)):
                        prefix = word[:i]
                        if prefix not in self.FREQ:
                            self.FREQ[prefix] = 0

        print(f"✅ 主词典加载完成！共计 {len(self.FREQ)} 个节点 (含前缀)。")

    def load_hmm(self, hmm_path):
        """动态加载训练好的 HMM JSON 模型"""
        print(f"⏳ 正在加载 HMM 未登录词识别模型: {hmm_path}")
        with open(hmm_path, 'r', encoding='utf-8') as f:
            model = json.load(f)
            self.start_p = model['START_P']
            self.trans_p = model['TRANS_P']
            self.emit_p = model['EMIT_P']
            self.hmm_enabled = True
        print(f"✅ HMM 模型挂载成功，随时准备对付生词！")

    def get_dag(self, sentence: str) -> dict:
        """为句子构建有向无环图（DAG）"""
        dag = {}
        N = len(sentence)
        for start in range(N):
            candidates = []
            end = start
            frag = sentence[start]
            while end < N and frag in self.FREQ:
                if self.FREQ[frag] > 0:
                    candidates.append(end)
                end += 1
                if end < N:
                    frag = sentence[start:end+1]
            if not candidates:
                candidates.append(start)
            dag[start] = candidates
        return dag

    def calc_dp(self, sentence: str, dag: dict) -> dict:
        """动态规划 (Dynamic Programming) 求解最大概率路径"""
        N = len(sentence)
        route = {}
        route[N] = (0.0, 0)
        
        log_total = math.log(self.total_freq) if self.total_freq > 0 else 1.0

        for idx in range(N - 1, -1, -1):
            candidates = []
            for end in dag[idx]:
                word = sentence[idx:end+1]
                freq = self.FREQ.get(word, 0)
                if freq <= 0:
                    freq = 1
                log_prob = math.log(freq) - log_total + route[end+1][0]
                candidates.append((log_prob, end))
            route[idx] = max(candidates, key=lambda item: item[0])
            
        return route

    def _cut_core(self, sentence: str) -> list:
        """
        核心分词逻辑：仅处理「纯中文」片段（无标点、无英文数字）。
        由 cut() 在预处理按标点/英文数字切分后对每段调用。
        """
        if not sentence:
            return []

        # 1. DAG + DP 基础切分
        dag = self.get_dag(sentence)
        route = self.calc_dp(sentence, dag)

        words = []
        pos = 0
        N = len(sentence)
        while pos < N:
            next_pos = route[pos][1] + 1
            words.append(sentence[pos:next_pos])
            pos = next_pos

        # 2. HMM 未登录词识别 (新词兜底聚合)
        final_words = []
        single_char_buf = ""

        def local_hmm_cut(text):
            if not self.hmm_enabled:
                return list(text)
            prob, pos_list = viterbi(text, STATES, self.start_p, self.trans_p, self.emit_p)
            begin, nexti = 0, 0
            res = []
            for i, char in enumerate(text):
                pos = pos_list[i]
                if pos == "B":
                    begin = i
                elif pos == "E":
                    res.append(text[begin : i + 1])
                    nexti = i + 1
                elif pos == "S":
                    res.append(char)
                    nexti = i + 1
            if nexti < len(text):
                res.append(text[nexti:])
            return res

        for word in words:
            if len(word) == 1:
                single_char_buf += word
            else:
                if single_char_buf:
                    if len(single_char_buf) == 1:
                        final_words.append(single_char_buf)
                    else:
                        final_words.extend(local_hmm_cut(single_char_buf))
                    single_char_buf = ""
                final_words.append(word)

        if single_char_buf:
            if len(single_char_buf) == 1:
                final_words.append(single_char_buf)
            else:
                final_words.extend(local_hmm_cut(single_char_buf))

        return final_words

    def cut(self, sentence: str) -> list:
        """
        执行混合分词。先按标点与英文/数字做硬截断与保护，再对纯中文段做 DAG+HMM 分词。
        - 英文、数字、C++ 等整块不参与分词，原样输出；
        - 标点作为天然边界，单独成 token。
        """
        if not sentence:
            return []
        segments = _tokenize_input(sentence)
        result = []
        for typ, content in segments:
            if typ == "chinese":
                result.extend(self._cut_core(content))
            elif typ == "en_num":
                result.append(content)
            else:  # punct
                result.append(content)
        return result

# ================= 测试入口 =================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dict_path = os.path.join(BASE_DIR, "data", "output_dict", "my_dict_primary.txt")

    segmenter = AutoSegmenter(dict_path)

    test_sentences = [
        "在这个从零开始的自然语言处理项目中，我们取得了巨大成功。",
        "这是一个测试，用来检验我们的分词效果到底好不好。",
        "今天天气真不错，我们一起去爬山吧。",
        "OpenClaw和PyTorch、Flutter都是技术名词。",
        "C++或F（混合）的写法要正确切分。",
        "（HMM）的、子'、了'、或'。",
    ]

    print("\n🔪 开始初步分词测试 (含英文/数字/标点):")
    print("=" * 50)
    for sent in test_sentences:
        words = segmenter.cut(sent)
        print(f"📝 原文: {sent}")
        print(f"✂️ 分词: {' / '.join(words)}")
        print("-" * 50)