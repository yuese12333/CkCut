from typing import List, Tuple

import torch
import torch.nn as nn

from .constants import START_TAG, STOP_TAG, TAG_TO_ID


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.tag_to_id = TAG_TO_ID
        self.tagset_size = len(self.tag_to_id)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 转移矩阵 transitions[to_tag, from_tag]
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[self.tag_to_id[START_TAG], :] = -10000.0
        self.transitions.data[:, self.tag_to_id[STOP_TAG]] = -10000.0

    def _get_lstm_features(self, sentences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        # 获取真实长度，供 LSTM 压缩序列使用
        lengths = masks.sum(dim=1).cpu()
        embeds = self.word_embeds(sentences)
        
        # 消除 PAD 对 LSTM 反向传播的干扰
        packed_embeds = nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed_embeds)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=sentences.size(1))
        
        feats = self.hidden2tag(self.dropout(lstm_out))
        return feats

    def _forward_alg(self, feats: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        B, L, C = feats.size()
        score = torch.full((B, C), -10000.0, device=feats.device)
        score[:, self.tag_to_id[START_TAG]] = 0.0

        for i in range(L):
            feat = feats[:, i, :] # (B, C)
            m = masks[:, i]       # (B,)

            # 核心张量运算：[B, 1, C] + [1, C, C] + [B, C, 1] -> [B, C, C]
            next_score = score.unsqueeze(1) + self.transitions.unsqueeze(0) + feat.unsqueeze(2)
            next_score = torch.logsumexp(next_score, dim=2) # 沿 from_tag 维度求和, 得到 [B, C]
            
            # 使用 Mask：如果是有效的字就更新 score，如果是 PAD 就保留原来的 score
            score = torch.where(m.unsqueeze(1), next_score, score)

        # 加上到 STOP_TAG 的转移概率
        score += self.transitions[self.tag_to_id[STOP_TAG]].unsqueeze(0)
        return torch.logsumexp(score, dim=1)

    def _score_sentence(self, feats: torch.Tensor, tags: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        B, L = tags.size()
        score = torch.zeros(B, device=feats.device)
        start_tags = torch.full((B, 1), self.tag_to_id[START_TAG], dtype=torch.long, device=tags.device)
        tags_ext = torch.cat([start_tags, tags], dim=1)

        for i in range(L):
            feat = feats[:, i, :]
            curr_tag = tags_ext[:, i+1]
            prev_tag = tags_ext[:, i]
            m = masks[:, i]

            emit_score = feat.gather(1, curr_tag.unsqueeze(1)).squeeze(1)
            trans_score = self.transitions[curr_tag, prev_tag]
            
            # 只有在 Mask 允许的情况下才累加得分
            score += (emit_score + trans_score) * m

        # 动态寻找每个句子最后一个真实有效标签的位置
        lengths = masks.sum(dim=1).long()
        last_tags = tags_ext.gather(1, lengths.unsqueeze(1)).squeeze(1)
        score += self.transitions[self.tag_to_id[STOP_TAG], last_tags]
        
        return score

    def _viterbi_decode(self, feats: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, List[List[int]]]:
        B, L, C = feats.size()
        score = torch.full((B, C), -10000.0, device=feats.device)
        score[:, self.tag_to_id[START_TAG]] = 0.0
        backpointers = []

        for i in range(L):
            feat = feats[:, i, :]
            m = masks[:, i]
            
            next_score = score.unsqueeze(1) + self.transitions.unsqueeze(0)
            next_score, bptrs = torch.max(next_score, dim=2)
            next_score += feat
            
            score = torch.where(m.unsqueeze(1), next_score, score)
            backpointers.append(bptrs)

        score += self.transitions[self.tag_to_id[STOP_TAG]].unsqueeze(0)
        best_scores, best_tags = torch.max(score, dim=1)

        # 一次性回传到 CPU，避免循环内频繁 .item() 造成同步阻塞
        lengths_cpu = masks.sum(dim=1).long().cpu().tolist()
        best_tags_cpu = best_tags.cpu().tolist()
        backpointers_cpu = torch.stack(backpointers).cpu().numpy()  # (L, B, C)

        best_paths: List[List[int]] = []
        for b in range(B):
            seq_len = lengths_cpu[b]
            if seq_len == 0:
                best_paths.append([])
                continue

            best_tag = best_tags_cpu[b]
            path = [best_tag]
            for i in range(seq_len - 1, 0, -1):
                best_tag = int(backpointers_cpu[i][b][best_tag])
                path.append(best_tag)
            path.reverse()
            best_paths.append(path)

        return best_scores, best_paths

    def neg_log_likelihood(self, sentences: torch.Tensor, tags: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        feats = self._get_lstm_features(sentences, masks)
        forward_score = self._forward_alg(feats, masks)
        gold_score = self._score_sentence(feats, tags, masks)
        # 返回整个 batch 的平均 Loss
        return torch.mean(forward_score - gold_score)

    def forward(self, sentences: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, List[List[int]]]:
        feats = self._get_lstm_features(sentences, masks)
        score, tag_seqs = self._viterbi_decode(feats, masks)
        return score, tag_seqs