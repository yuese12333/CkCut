import json
import random
import sys
from dataclasses import dataclass, asdict
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from .data_pipeline import (
    SegmentedDataset,
    build_char_vocab,
    collate_batch,
    encode_chars,
    read_segmented_files,
    save_vocab,
)
from .model import BiLSTMCRF


@dataclass
class TrainConfig:
    train_dir: str
    output_dir: str
    embedding_dim: int = 128
    hidden_dim: int = 256
    lr: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 5
    batch_size: int = 8
    min_char_freq: int = 1
    max_samples: int = 0
    seed: int = 42
    device: str = "auto"
    num_workers: int = 2
    pin_memory: bool = True
    use_amp: bool = True
    prefetch_factor: int = 2
    clip_grad_norm: float = 5.0 
    bucket_multiplier: int = 100
    log_interval: int = 20


class LengthBucketBatchSampler(Sampler[List[int]]):
    """按句长分桶后再组 batch，降低 padding 浪费。"""

    def __init__(self, lengths: List[int], batch_size: int, bucket_size: int, shuffle: bool = True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_size = max(bucket_size, batch_size)
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(indices)

        batches: List[List[int]] = []
        for i in range(0, len(indices), self.bucket_size):
            bucket = indices[i : i + self.bucket_size]
            bucket.sort(key=lambda idx: self.lengths[idx])
            for j in range(0, len(bucket), self.batch_size):
                batch = bucket[j : j + self.batch_size]
                if batch:
                    batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)
        for b in batches:
            yield b

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _prepare_samples(
    train_dir: str,
    min_char_freq: int,
    max_samples: int,
) -> Tuple[Dict[str, int], List[Tuple[List[int], List[int]]]]:
    raw_samples = list(read_segmented_files(train_dir))
    if max_samples > 0:
        raw_samples = raw_samples[:max_samples]
    if not raw_samples:
        raise ValueError("训练样本为空。")

    char_to_id = build_char_vocab(raw_samples, min_freq=min_char_freq)
    encoded = [(encode_chars(chars, char_to_id), tags) for chars, tags in raw_samples]
    return char_to_id, encoded


def train_model(cfg: TrainConfig) -> Dict[str, float]:
    _set_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    char_to_id, encoded_samples = _prepare_samples(cfg.train_dir, cfg.min_char_freq, cfg.max_samples)
    dataset = SegmentedDataset(encoded_samples)
    lengths = [len(x) for x, _ in encoded_samples]
    bucket_size = cfg.batch_size * cfg.bucket_multiplier
    batch_sampler = LengthBucketBatchSampler(
        lengths=lengths,
        batch_size=cfg.batch_size,
        bucket_size=bucket_size,
        shuffle=True,
    )

    loader_kwargs = {
        "dataset": dataset,
        "batch_sampler": batch_sampler,
        "collate_fn": collate_batch,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory and use_cuda,
    }
    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg.prefetch_factor
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(**loader_kwargs)

    model = BiLSTMCRF(
        vocab_size=len(char_to_id),
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
    ).to(device)
    # Linux/WSL 下可尝试 graph compile 获取额外提速；Windows 先保守关闭。
    if hasattr(torch, "compile") and sys.platform != "win32":
        model = torch.compile(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # 去掉了废弃的 verbose=True 消除警告
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_vocab(str(out_dir / "char_vocab.json"), char_to_id)

    best_loss = float("inf")
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        sent_count = 0
        recent_losses = deque(maxlen=100)
        
        current_lr = optimizer.param_groups[0]['lr']
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs} [LR: {current_lr:.2e}]", unit="batch")
        
        # 【修正的核心部位】直接用 3 个变量接收 collate_batch 的返回值
        for step, (sentences, tags, masks) in enumerate(pbar, start=1):
            optimizer.zero_grad(set_to_none=True)
            amp_dtype = torch.bfloat16 if use_cuda else torch.float32
            
            sentences = sentences.to(device, non_blocking=use_cuda)
            tags = tags.to(device, non_blocking=use_cuda)
            masks = masks.to(device, non_blocking=use_cuda)
            
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(cfg.use_amp and use_cuda)):
                # 在 model.py 中已经重写了这里的逻辑，直接算出一个 batch 的平均 loss
                loss = model.neg_log_likelihood(sentences, tags, masks)
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            # 更新参数
            optimizer.step()
            
            # 更新统计信息
            batch_size_actual = sentences.size(0)
            running += loss.item() * batch_size_actual
            sent_count += batch_size_actual
            recent_losses.append(loss.item())
            
            if step % max(cfg.log_interval, 1) == 0:
                recent_avg = sum(recent_losses) / len(recent_losses)
                pbar.set_postfix(avg_loss=f"{recent_avg:.4f}")

        epoch_loss = running / max(sent_count, 1)
        history.append({"epoch": epoch, "avg_sentence_loss": epoch_loss})

        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), out_dir / "bilstm_crf.pt")

    with open(out_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "history": history}, f, ensure_ascii=False, indent=2)

    return {
        "best_loss": best_loss,
        "samples": len(encoded_samples),
        "vocab_size": len(char_to_id),
        "device": str(device),
        "amp_enabled": bool(cfg.use_amp and use_cuda),
    }