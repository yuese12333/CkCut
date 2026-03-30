"""将训练得到的 BiLSTM-CRF 权重导出为 ONNX（BiLSTM+发射层）+ CRF 转移矩阵元数据。"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .model import BiLSTMCRF


def _dims_from_state_dict(state: dict) -> Tuple[int, int]:
    """从权重张量推断 embedding_dim、hidden_dim（与 train_history 无关，避免导出时写错维度）。"""
    emb = int(state["word_embeds.weight"].shape[1])
    hidden = int(state["hidden2tag.weight"].shape[1])
    return emb, hidden


class BiLSTMEmission(nn.Module):
    """与训练时一致的前向：整段 pad 输入 LSTM（与 pack 版在「无 pad」时数值一致；batch 内有 pad 时可能略有差异）。"""

    def __init__(self, crf: BiLSTMCRF):
        super().__init__()
        self.word_embeds = crf.word_embeds
        self.lstm = crf.lstm
        self.dropout = crf.dropout
        self.hidden2tag = crf.hidden2tag

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        emb = self.word_embeds(token_ids)
        out, _ = self.lstm(emb)
        return self.hidden2tag(self.dropout(out))


def export_bilstm_crf_onnx(
    model_path: str,
    vocab_path: str,
    out_onnx: str,
    embedding_dim: int,
    hidden_dim: int,
    opset: int = 17,
) -> Tuple[str, str]:
    """
    导出 `out_onnx`（默认 .onnx）及同目录 `*_meta.json`（含转移矩阵）。
    返回 (onnx_path, meta_path)。

    embedding_dim / hidden_dim 参数会被忽略，实际维度始终从 .pt 的 state_dict 推断，
    与训练时 train_history 一致，避免手工填错。
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    try:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location="cpu")
    inf_e, inf_h = _dims_from_state_dict(state)
    embedding_dim, hidden_dim = inf_e, inf_h

    model = BiLSTMCRF(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    model.load_state_dict(state)
    model.eval()

    emitter = BiLSTMEmission(model).eval()

    # dummy 用 batch=1、seq=1；dynamic_axes 仍允许推理时任意 batch / 序列长。
    # PyTorch 会对「动态 batch + LSTM」打印保守提示，ORT 实际多可正常运行，此处抑制该条。
    dummy = torch.zeros((1, 1), dtype=torch.long)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Exporting a model to ONNX with a batch_size other than 1.*",
            category=UserWarning,
        )
        torch.onnx.export(
            emitter,
            (dummy,),
            out_onnx,
            input_names=["token_ids"],
            output_names=["emissions"],
            dynamic_axes={
                "token_ids": {0: "batch", 1: "seq_len"},
                "emissions": {0: "batch", 1: "seq_len"},
            },
            opset_version=opset,
        )

    trans = model.transitions.detach().cpu().numpy().astype(np.float32)
    meta_path = os.path.splitext(out_onnx)[0] + "_meta.json"
    meta = {
        "format_version": 1,
        "vocab_path_hint": os.path.basename(vocab_path),
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "opset": opset,
        "transitions": trans.tolist(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return out_onnx, meta_path


def main():
    p = argparse.ArgumentParser(description="导出 BiLSTM-CRF 为 ONNX + CRF meta（无需部署 PyTorch）")
    p.add_argument("--model_path", required=True, help=".pt 权重路径")
    p.add_argument("--vocab_path", required=True, help="char_vocab.json")
    p.add_argument("--out_onnx", default="", help="输出 .onnx；默认与 .pt 同目录同名")
    p.add_argument("--embedding_dim", type=int, default=128, help="已废弃，维度从权重自动推断")
    p.add_argument("--hidden_dim", type=int, default=256, help="已废弃，维度从权重自动推断")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    out_onnx = args.out_onnx
    if not out_onnx:
        root, _ = os.path.splitext(args.model_path)
        out_onnx = root + ".onnx"

    onnx_p, meta_p = export_bilstm_crf_onnx(
        args.model_path,
        args.vocab_path,
        out_onnx,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        opset=args.opset,
    )
    print("已写入:", onnx_p)
    print("已写入:", meta_p)


if __name__ == "__main__":
    main()
