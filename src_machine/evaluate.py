import os
from pathlib import Path
from tqdm import tqdm

from src_machine.segmenter import AutoSegmenter


class Evaluator:
    @staticmethod
    def get_word_spans(words: list) -> set:
        spans = set()
        offset = 0
        for word in words:
            length = len(word)
            spans.add((offset, offset + length))
            offset += length
        return spans

    @staticmethod
    def run(segmenter: AutoSegmenter, test_dir: str):
        test_path = Path(test_dir)
        if not test_path.exists() or not test_path.is_dir():
            print(f"Error: test dir not found: {test_dir}")
            return

        txt_files = list(test_path.rglob("*.txt"))
        if not txt_files:
            print(f"Warning: no .txt found in {test_dir}")
            return

        total_gold_words = 0
        total_pred_words = 0
        total_correct_words = 0

        print(f"Evaluating on {test_dir} ({len(txt_files)} files)...")

        with tqdm(txt_files, desc="Eval", unit="file") as pbar:
            for file_path in pbar:
                pbar.set_postfix(file=file_path.name)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        gold_words = line.split()
                        if not gold_words:
                            continue

                        raw_sentence = "".join(gold_words)
                        pred_words = segmenter.cut(raw_sentence)

                        gold_spans = Evaluator.get_word_spans(gold_words)
                        pred_spans = Evaluator.get_word_spans(pred_words)
                        correct_spans = gold_spans.intersection(pred_spans)

                        total_gold_words += len(gold_spans)
                        total_pred_words += len(pred_spans)
                        total_correct_words += len(correct_spans)

        if total_pred_words == 0 or total_gold_words == 0:
            print("Warning: empty prediction or empty gold.")
            return

        precision = total_correct_words / total_pred_words
        recall = total_correct_words / total_gold_words
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print("\n" + "=" * 40)
        print("Global Evaluation")
        print("=" * 40)
        print(f"Files: {len(txt_files)}")
        print(f"Gold words: {total_gold_words}")
        print(f"Pred words: {total_pred_words}")
        print(f"Correct words: {total_correct_words}")
        print("-" * 40)
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall:    {recall * 100:.2f}%")
        print(f"F1:        {f1_score * 100:.2f}%")
        print("=" * 40)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dict_path = os.path.join(base_dir, "data", "output_dict", "my_dict_wiki.txt")
    hmm_path = os.path.join(base_dir, "data", "output_dict", "hmm_model.json")
    test_dir_path = os.path.join(base_dir, "data", "evaluate")

    segmenter = AutoSegmenter(dict_path, hmm_path)
    Evaluator.run(segmenter, test_dir_path)
