import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .constants import PAD_CHAR, TAG_TO_ID, UNK_CHAR, split_han_blocks


def words_to_bmes(words: Sequence[str]) -> Tuple[List[str], List[int]]:
    chars: List[str] = []
    tags: List[int] = []
    for word in words:
        if not word:
            continue
        if len(word) == 1:
            chars.append(word)
            tags.append(TAG_TO_ID["S"])
            continue
        chars.extend(word)
        tags.append(TAG_TO_ID["B"])
        for _ in range(len(word) - 2):
            tags.append(TAG_TO_ID["M"])
        tags.append(TAG_TO_ID["E"])
    return chars, tags


def read_segmented_files(data_dir_or_file: str) -> Iterable[Tuple[List[str], List[int]]]:
    base = Path(data_dir_or_file)
    if not base.exists():
        raise FileNotFoundError(f"训练数据路径不存在: {data_dir_or_file}")

    # 支持传入单个 .txt 文件（用于“每个训练文件分别训练”）
    if base.is_file():
        if base.suffix.lower() != ".txt":
            raise ValueError(f"只支持 .txt 训练文件，实际为: {base}")
        txt_files = [base]
    else:
        txt_files = sorted(base.rglob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"训练数据目录下没有 txt 文件: {data_dir_or_file}")

    for file_path in txt_files:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                words = line.strip().split()
                if not words:
                    continue
                chars, tags = words_to_bmes(words)
                if chars and len(chars) == len(tags):
                    yield chars, tags


def build_char_vocab(samples: Iterable[Tuple[List[str], List[int]]], min_freq: int = 1) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for chars, _ in samples:
        for ch in chars:
            freq[ch] = freq.get(ch, 0) + 1

    vocab = {PAD_CHAR: 0, UNK_CHAR: 1}
    for ch, count in sorted(freq.items(), key=lambda kv: kv[1], reverse=True):
        if count >= min_freq and ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def encode_chars(chars: Sequence[str], char_to_id: Dict[str, int]) -> List[int]:
    unk = char_to_id[UNK_CHAR]
    return [char_to_id.get(ch, unk) for ch in chars]


class SegmentedDataset(Dataset):
    def __init__(self, encoded_samples: List[Tuple[List[int], List[int]]]):
        self.samples = encoded_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    真正的批处理收集器：
    1. 将句子用 PAD 补齐到等长
    2. 生成对应的 bool Mask 矩阵
    """
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # 假设 PAD_CHAR 对应的 ID 是 0 (在 build_char_vocab 中定义的)
    PAD_ID = 0
    
    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=PAD_ID)
    padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=PAD_ID)
    
    # 生成掩码，True 表示有效字符，False 表示 PAD
    masks = (padded_inputs != PAD_ID)
    
    return padded_inputs, padded_targets, masks


def save_vocab(path: str, char_to_id: Dict[str, int]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(char_to_id, f, ensure_ascii=False, indent=2)


def load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def line_to_char_ids(raw_line: str, char_to_id: Dict[str, int]) -> List[List[int]]:
    blocks = split_han_blocks(raw_line.strip())
    return [encode_chars(list(block), char_to_id) for block in blocks if block]
