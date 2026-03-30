from typing import Dict, List

import torch

from .constants import ID_TO_TAG
from .vocab_io import encode_chars, load_vocab
from .model import BiLSTMCRF


class CRFSegmenter:
    def __init__(self, model_path: str, vocab_path: str, embedding_dim: int, hidden_dim: int, device: str = "cpu"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        self.char_to_id: Dict[str, int] = load_vocab(vocab_path)
        self.model = BiLSTMCRF(
            vocab_size=len(self.char_to_id),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        # 优先使用 weights_only=True 以避免未来版本的安全警告；
        # 旧版 PyTorch 不支持该参数时回退到原调用。
        try:
            state = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def _bmes_to_words(self, chars: List[str], tag_ids: List[int]) -> List[str]:
        tags = [ID_TO_TAG[i] for i in tag_ids]
        words = []
        buffer = ""
        for ch, tag in zip(chars, tags):
            if tag == "S":
                if buffer:
                    words.append(buffer)
                    buffer = ""
                words.append(ch)
            elif tag == "B":
                if buffer:
                    words.append(buffer)
                buffer = ch
            elif tag == "M":
                buffer += ch
            elif tag == "E":
                buffer += ch
                words.append(buffer)
                buffer = ""
            else:
                if buffer:
                    words.append(buffer)
                    buffer = ""
                words.append(ch)
        if buffer:
            words.append(buffer)
        return words

    def cut(self, text: str) -> List[str]:
        if not text.strip():
            return []
        chars = list(text.strip())
        ids = encode_chars(chars, self.char_to_id)
        
        # 增加 batch 维度：(1, L)
        sent = torch.tensor([ids], dtype=torch.long, device=self.device)
        # 生成全 True 的 mask：(1, L)
        mask = torch.ones_like(sent, dtype=torch.bool, device=self.device)
        
        with torch.inference_mode():
            _, tag_seqs = self.model(sent, mask)
            
        # tag_seqs 是一个列表的列表，取第一个 batch_item
        return self._bmes_to_words(chars, tag_seqs[0])

    def cut_batch(self, texts: List[str], batch_size: int = 64) -> List[List[str]]:
        """批量分词，显著减少 batch_size=1 时的 GPU 调度开销。"""
        results: List[List[str]] = []
        if not texts:
            return results

        pad_id = 0
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_chars: List[List[str]] = []
            batch_ids: List[List[int]] = []
            max_len = 0

            for text in batch_texts:
                chars = list(text.strip())
                batch_chars.append(chars)
                if not chars:
                    batch_ids.append([])
                    continue
                ids = encode_chars(chars, self.char_to_id)
                batch_ids.append(ids)
                max_len = max(max_len, len(ids))

            if max_len == 0:
                results.extend([[] for _ in batch_texts])
                continue

            padded_ids: List[List[int]] = []
            masks: List[List[bool]] = []
            for ids in batch_ids:
                pad_len = max_len - len(ids)
                padded_ids.append(ids + [pad_id] * pad_len)
                masks.append([True] * len(ids) + [False] * pad_len)

            sent_tensor = torch.tensor(padded_ids, dtype=torch.long, device=self.device)
            mask_tensor = torch.tensor(masks, dtype=torch.bool, device=self.device)

            with torch.inference_mode():
                _, tag_seqs = self.model(sent_tensor, mask_tensor)

            for chars, tag_seq in zip(batch_chars, tag_seqs):
                if not chars:
                    results.append([])
                else:
                    results.append(self._bmes_to_words(chars, tag_seq))

        return results
