import os
import sys
import argparse
from src_machine.preprocess import preprocess_directory
from src_machine.word_discovery import WordDiscoverer
from src_machine.segmenter import AutoSegmenter
from src_machine.evaluate import Evaluator
from src_machine.hmm_trainer import HMMTrainer


def main():
    # Windows console may default to GBK, force UTF-8 output.
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", errors="replace")
    if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
        sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="AutoSeg: DAG+HMM Chinese segmentation pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--step",
        type=str,
        required=True,
        help="""Pipeline steps combination:
A: preprocess raw_corpus
B: word discovery and dict generation
C: HMM supervised training
D: evaluation (P/R/F1)
E: interactive segmentation
all: A -> B -> C -> D -> E
ABCD: A -> B -> C -> D (no interactive E)""",
    )

    parser.add_argument("--max_len", type=int, default=4, help="max candidate word length")
    parser.add_argument("--min_freq", type=int, default=5, help="min frequency threshold")
    parser.add_argument("--min_pmi", type=float, default=4.0, help="min PMI threshold")
    parser.add_argument("--min_entropy", type=float, default=1.0, help="min entropy threshold")
    parser.add_argument("--withouthmm", action="store_true", help="disable HMM fallback")

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_corpus_dir = os.path.join(base_dir, "data", "raw_corpus")
    processed_file = os.path.join(base_dir, "data", "processed", "merged_cleaned.txt")
    output_dict = os.path.join(base_dir, "data", "output_dict", "my_dict_wiki.txt")

    hmm_train_dir = os.path.join(base_dir, "data", "HMM_train")
    hmm_model = os.path.join(base_dir, "data", "output_dict", "hmm_model.json")

    test_dir = os.path.join(base_dir, "data", "evaluate")

    step_cmd = args.step.upper()
    abcd_only = step_cmd == "ABCD"

    run_a = "A" in step_cmd or step_cmd == "ALL"
    run_b = "B" in step_cmd or step_cmd == "ALL"
    run_c = "C" in step_cmd or step_cmd == "ALL"
    run_d = "D" in step_cmd or step_cmd == "ALL"
    run_e = ("E" in step_cmd or step_cmd == "ALL") and not abcd_only

    if not any([run_a, run_b, run_c, run_d, run_e]):
        print("Invalid --step. Use A/B/C/D/E/all/ABCD.")
        return

    if run_a:
        print("\n" + "=" * 50)
        print("[A] Preprocess corpus")
        print("=" * 50)
        preprocess_directory(raw_corpus_dir, processed_file)

    if run_b:
        print("\n" + "=" * 50)
        print("[B] Word discovery")
        print("=" * 50)
        if not os.path.exists(processed_file):
            print(f"Missing processed corpus: {processed_file}. Please run step A first.")
            return

        discoverer = WordDiscoverer(max_word_len=args.max_len, min_freq=args.min_freq)
        discoverer.count_ngrams(processed_file)
        discoverer.compute_pmi(min_pmi=args.min_pmi)
        discoverer.compute_entropy(processed_file, min_entropy=args.min_entropy)

        print(f"Exporting dictionary to: {output_dict}")
        sorted_words = sorted(discoverer.final_words.items(), key=lambda x: x[1], reverse=True)

        dir_name = os.path.dirname(output_dict)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(output_dict, "w", encoding="utf-8") as f:
            for word, freq in sorted_words:
                f.write(f"{word} {freq}\n")
        print("Dictionary saved.")

    if run_c:
        print("\n" + "=" * 50)
        print("[C] HMM training")
        print("=" * 50)
        if not os.path.exists(hmm_train_dir) or not os.path.isdir(hmm_train_dir):
            print(f"Missing HMM train dir: {hmm_train_dir}")
            return

        trainer = HMMTrainer()
        trainer.train(hmm_train_dir)
        trainer.save_model(hmm_model)

    if run_d:
        print("\n" + "=" * 50)
        print("[D] Evaluation")
        print("=" * 50)
        if not os.path.exists(output_dict):
            print(f"Missing dictionary: {output_dict}. Please run step B first.")
            return

        if not os.path.exists(test_dir) or not os.path.isdir(test_dir):
            print(f"Missing test dir: {test_dir}")
        else:
            active_hmm = None if args.withouthmm else hmm_model
            if args.withouthmm:
                print("Running pure DAG mode (--withouthmm)")
            segmenter = AutoSegmenter(output_dict, active_hmm)
            Evaluator.run(segmenter, test_dir)

    if run_e:
        print("\n" + "=" * 50)
        print("[E] Interactive segmentation")
        print("=" * 50)
        if not os.path.exists(output_dict):
            print(f"Missing dictionary: {output_dict}. Please run step B first.")
            return

        active_hmm = None if args.withouthmm else hmm_model
        segmenter = AutoSegmenter(output_dict, active_hmm)

        mode_text = "Pure DAG" if args.withouthmm else "DAG+HMM"
        print(f"{mode_text} ready. Input text (q/exit to quit).")

        while True:
            try:
                text = input("\nInput: ").strip()
                if text.lower() in ["q", "exit"]:
                    print("Bye.")
                    break
                if not text:
                    continue
                words = segmenter.cut(text)
                print("Result:", " / ".join(words))
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break


if __name__ == "__main__":
    main()
