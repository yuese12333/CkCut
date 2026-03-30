"""词表读写与编码（无 PyTorch 依赖，供 ONNX 推理等使用）。"""

import json
from pathlib import Path
from typing import Dict, List, Sequence

from .constants import UNK_CHAR


def load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_vocab(path: str, char_to_id: Dict[str, int]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(char_to_id, f, ensure_ascii=False, indent=2)


def encode_chars(chars: Sequence[str], char_to_id: Dict[str, int]) -> List[int]:
    unk = char_to_id[UNK_CHAR]
    return [char_to_id.get(ch, unk) for ch in chars]
