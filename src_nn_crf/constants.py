import re
from typing import List

START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD_CHAR = "<PAD>"
UNK_CHAR = "<UNK>"

TAGS = ["B", "M", "E", "S", START_TAG, STOP_TAG]
TAG_TO_ID = {tag: idx for idx, tag in enumerate(TAGS)}
ID_TO_TAG = {idx: tag for tag, idx in TAG_TO_ID.items()}

CHINESE_BLOCK = re.compile(r"[\u4e00-\u9fff]+")


def split_han_blocks(text: str) -> List[str]:
    return CHINESE_BLOCK.findall(text)
