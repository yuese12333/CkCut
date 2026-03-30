"""CRF Viterbi 解码（NumPy），与 `BiLSTMCRF._viterbi_decode` 逻辑一致。"""

from typing import List

import numpy as np

from .constants import START_TAG, STOP_TAG, TAG_TO_ID


def viterbi_decode_batch(emissions: np.ndarray, transitions: np.ndarray, mask: np.ndarray) -> List[List[int]]:
    """
    emissions: float32/float64, shape [B, L, C]
    transitions: shape [C, C], transitions[to_tag, from_tag]（与 PyTorch 模块一致）
    mask: bool, shape [B, L]，True 表示有效时间步
    """
    start_i = TAG_TO_ID[START_TAG]
    stop_i = TAG_TO_ID[STOP_TAG]
    B, L, C = emissions.shape
    neg = -10000.0
    all_paths: List[List[int]] = []

    for b in range(B):
        score = np.full((C,), neg, dtype=np.float64)
        score[start_i] = 0.0
        backpointers: List[np.ndarray] = []

        for i in range(L):
            feat = emissions[b, i, :].astype(np.float64, copy=False)
            if not mask[b, i]:
                backpointers.append(np.zeros(C, dtype=np.int64))
                continue

            # S[to, from] = score[from] + transitions[to, from]
            s = score[np.newaxis, :] + transitions
            bptrs = np.argmax(s, axis=1).astype(np.int64)
            next_score = np.max(s, axis=1) + feat
            score = next_score
            backpointers.append(bptrs)

        score = score + transitions[stop_i, :]
        best_tag = int(np.argmax(score))
        seq_len = int(mask[b].sum())
        if seq_len == 0:
            all_paths.append([])
            continue

        path = [best_tag]
        tag = best_tag
        for i in range(seq_len - 1, 0, -1):
            tag = int(backpointers[i][tag])
            path.append(tag)
        path.reverse()
        all_paths.append(path)

    return all_paths
