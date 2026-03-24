from pathlib import Path
from typing import Dict

from .infer import CRFSegmenter


def _spans(words):
    s = set()
    offset = 0
    for w in words:
        s.add((offset, offset + len(w)))
        offset += len(w)
    return s


def evaluate(seg: CRFSegmenter, test_dir: str, batch_size: int = 128) -> Dict[str, float]:
    base = Path(test_dir)
    if not base.exists():
        raise FileNotFoundError(f"评测目录不存在: {test_dir}")
    txt_files = sorted(base.rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"评测目录中没有 txt 文件: {test_dir}")

    total_gold = 0
    total_pred = 0
    total_correct = 0

    gold_sentences = []
    raw_sentences = []

    for fp in txt_files:
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                gold_words = line.strip().split()
                if not gold_words:
                    continue
                gold_sentences.append(gold_words)
                raw_sentences.append("".join(gold_words))

    pred_sentences = seg.cut_batch(raw_sentences, batch_size=batch_size)
    for gold_words, pred_words in zip(gold_sentences, pred_sentences):
        g = _spans(gold_words)
        p = _spans(pred_words)
        total_gold += len(g)
        total_pred += len(p)
        total_correct += len(g.intersection(p))

    precision = total_correct / total_pred if total_pred else 0.0
    recall = total_correct / total_gold if total_gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_gold": total_gold,
        "total_pred": total_pred,
        "total_correct": total_correct,
    }
