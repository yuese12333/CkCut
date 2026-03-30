import argparse
import json
import os

def build_parser():
    p = argparse.ArgumentParser(description="BiLSTM-CRF 中文分词全新流程")
    p.add_argument("--mode", required=True, choices=["train", "eval", "infer", "export_onnx"])

    p.add_argument("--train_dir", default="data/HMM_train")
    p.add_argument("--test_dir", default="data/evaluate")
    p.add_argument("--output_dir", default="data/output_nn_crf")
    p.add_argument("--model_path", default="data/output_nn_crf/bilstm_crf.pt")
    p.add_argument("--vocab_path", default="data/output_nn_crf/char_vocab.json")

    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--min_char_freq", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=0, help="训练样本上限，0表示全量")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader 预取批次数（num_workers>0 时生效）")
    p.add_argument("--pin_memory", action="store_true", help="训练 DataLoader 开启 pin_memory")
    p.add_argument("--use_amp", action="store_true", help="CUDA 上启用自动混合精度")
    p.add_argument("--clip_grad_norm", type=float, default=5.0, help="梯度裁剪阈值")
    p.add_argument("--omp_threads", type=int, default=1, help="CPU/OpenMP 线程数")
    p.add_argument("--train_separately", action="store_true", help="训练时对 train_dir 下每个 .txt 分别训练并分别输出权重")
    p.add_argument(
        "--out_onnx",
        default="",
        help="mode=export_onnx 时输出 .onnx 路径；默认同目录下与 .pt 同名（如 bilstm_crf.onnx），并写入 *_meta.json",
    )
    p.add_argument("--onnx_opset", type=int, default=17, help="export_onnx 使用的 ONNX opset")
    return p


def _abs(root: str, maybe_rel: str) -> str:
    if os.path.isabs(maybe_rel):
        return maybe_rel
    return os.path.join(root, maybe_rel)


def main():
    # Windows 下部分科学计算依赖会重复加载 OpenMP 运行时，先注入兼容开关再导入 torch 相关模块。
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    args = build_parser().parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(args.omp_threads))

    from src_nn_crf import CRFSegmenter, TrainConfig, evaluate, train_model

    root = os.path.dirname(os.path.abspath(__file__))

    train_dir = _abs(root, args.train_dir)
    test_dir = _abs(root, args.test_dir)
    output_dir = _abs(root, args.output_dir)
    model_path = _abs(root, args.model_path)
    vocab_path = _abs(root, args.vocab_path)

    if args.mode == "export_onnx":
        from src_nn_crf.export_onnx import export_bilstm_crf_onnx

        out_onnx = args.out_onnx.strip()
        if not out_onnx:
            root_pt, _ = os.path.splitext(model_path)
            out_onnx = root_pt + ".onnx"
        else:
            out_onnx = _abs(root, out_onnx)
        onnx_p, meta_p = export_bilstm_crf_onnx(
            model_path,
            vocab_path,
            out_onnx,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            opset=args.onnx_opset,
        )
        print("ONNX:", onnx_p)
        print("META:", meta_p)
        return

    if args.mode == "train":
        if args.train_separately:
            from pathlib import Path
            base = Path(train_dir)
            if not base.exists():
                raise FileNotFoundError(f"训练目录不存在: {train_dir}")
            if base.is_file():
                raise ValueError("--train_separately 需要传入目录路径，而不是单个文件。")

            txt_files = sorted([p for p in base.glob("*.txt") if p.is_file()])
            if not txt_files:
                raise FileNotFoundError(f"目录内没有找到 .txt 训练文件: {train_dir}")

            for file_path in txt_files:
                one_out_dir = os.path.join(output_dir, file_path.stem)
                print(f"\n========== 开始训练: {file_path.name} ==========")
                cfg = TrainConfig(
                    train_dir=str(file_path),
                    output_dir=one_out_dir,
                    embedding_dim=args.embedding_dim,
                    hidden_dim=args.hidden_dim,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    min_char_freq=args.min_char_freq,
                    max_samples=args.max_samples,
                    seed=args.seed,
                    device=args.device,
                    num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor,
                    pin_memory=args.pin_memory or (args.device in {"auto", "cuda"}),
                    use_amp=args.use_amp or (args.device in {"auto", "cuda"}),
                    clip_grad_norm=args.clip_grad_norm,
                )
                stats = train_model(cfg)
                print(f"{file_path.name} 训练完成:", json.dumps(stats, ensure_ascii=False))
            return

        cfg = TrainConfig(
            train_dir=train_dir,
            output_dir=output_dir,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
            min_char_freq=args.min_char_freq,
            max_samples=args.max_samples,
            seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory or (args.device in {"auto", "cuda"}),
            use_amp=args.use_amp or (args.device in {"auto", "cuda"}),
            clip_grad_norm=args.clip_grad_norm,
        )
        stats = train_model(cfg)
        print("训练完成:", json.dumps(stats, ensure_ascii=False))
        return

    seg = CRFSegmenter(
        model_path=model_path,
        vocab_path=vocab_path,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )

    if args.mode == "eval":
        metrics = evaluate(seg, test_dir)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    while True:
        text = input("请输入句子（q退出）: ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            break
        print(" / ".join(seg.cut(text)))


if __name__ == "__main__":
    main()
