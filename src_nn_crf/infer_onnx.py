"""使用 ONNX Runtime 推理 BiLSTM 发射分数 + NumPy Viterbi，不依赖 PyTorch。"""

import json
from typing import Dict, List, Optional

import numpy as np

from .constants import ID_TO_TAG
from .vocab_io import encode_chars, load_vocab
from .viterbi_numpy import viterbi_decode_batch


class OnnxCRFSegmenter:
    def __init__(self, onnx_path: str, meta_path: str, vocab_path: str, providers: Optional[List[str]] = None):
        import onnxruntime as ort

        self.char_to_id: Dict[str, int] = load_vocab(vocab_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.transitions = np.array(meta["transitions"], dtype=np.float32)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=providers or ort.get_available_providers(),
        )
        self._input_name = self._session.get_inputs()[0].name

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
        sent = np.array([ids], dtype=np.int64)
        emissions = self._session.run(None, {self._input_name: sent})[0].astype(np.float32)
        mask = np.ones((1, len(ids)), dtype=bool)
        tag_seqs = viterbi_decode_batch(emissions, self.transitions, mask)
        return self._bmes_to_words(chars, tag_seqs[0])

    def cut_batch(self, texts: List[str], batch_size: int = 64) -> List[List[str]]:
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
                pl = max_len - len(ids)
                padded_ids.append(ids + [pad_id] * pl)
                masks.append([True] * len(ids) + [False] * pl)
            sent = np.array(padded_ids, dtype=np.int64)
            mask_arr = np.array(masks, dtype=bool)
            emissions = self._session.run(None, {self._input_name: sent})[0].astype(np.float32)
            tag_seqs = viterbi_decode_batch(emissions, self.transitions, mask_arr)
            for chars, tag_seq in zip(batch_chars, tag_seqs):
                if not chars:
                    results.append([])
                else:
                    results.append(self._bmes_to_words(chars, tag_seq))
        return results
