import os
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import QProcess, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QProgressBar,
)


def _safe_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except Exception:
        return default


class NNCRFTrainEvalWindow(QMainWindow):
    """
    BiLSTM-CRF 训练 + 评估一体化可视化页面。

    关键点：
    1) 语料库：支持单选（下拉）或多选（列表勾选）
    2) 每个语料训练完成后自动评估，并将 P/R/F1 填入表格
    3) 训练过程使用 QProcess 子进程，GUI 不阻塞；日志实时追加
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚀 CkCut NN-CRF 训练 + 评估一体化")
        self.resize(980, 720)

        self.process: Optional[QProcess] = None
        self.stage: Optional[str] = None  # "train" | "eval"
        self.current_corpus_stem: Optional[str] = None
        self.metrics_by_stem: Dict[str, Dict[str, float]] = {}

        self._work_dir = os.path.dirname(os.path.abspath(__file__))
        self.main_py = os.path.join(self._work_dir, "main_nn_crf.py")
        self.train_dir = os.path.join(self._work_dir, "data", "HMM_train")
        self.test_dir = os.path.join(self._work_dir, "data", "evaluate")
        self.default_output_dir = os.path.join(self._work_dir, "data", "output_nn_crf_gui")

        if not os.path.exists(self.main_py):
            raise FileNotFoundError(f"找不到入口文件: {self.main_py}")

        self._corpus_files = self._discover_corpora()
        if not self._corpus_files:
            raise FileNotFoundError(f"训练语料目录为空或不存在: {self.train_dir}")

        self._init_ui()

    def _discover_corpora(self) -> Dict[str, str]:
        base = Path(self.train_dir)
        if not base.exists():
            return {}
        files = {}
        for fp in sorted(base.glob("*.txt")):
            files[fp.stem] = str(fp)
        return files

    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # ---------- 顶部：参数 ----------
        param_group = QGroupBox("训练参数设置")
        param_layout = QGridLayout()

        row = 0

        self.cmb_preset = QComboBox()
        self.cmb_preset.addItems(
            [
                "Debug CPU（跑通流程）",
                "GPU 平衡（推荐）",
                "GPU 快速（更快但更稳）",
                "GPU 保守（防 OOM）",
            ]
        )
        param_layout.addWidget(QLabel("预设:"), row, 0)
        param_layout.addWidget(self.cmb_preset, row, 1, 1, 3)
        row += 1

        left_row = row
        right_row = row

        self.cmb_device = QComboBox()
        self.cmb_device.addItems(["cuda", "cpu", "auto"])
        self.cmb_device.setCurrentText("cuda")
        param_layout.addWidget(QLabel("device:"), left_row, 0)
        param_layout.addWidget(self.cmb_device, left_row, 1)
        left_row += 1

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 100)
        self.spin_epochs.setValue(10)
        param_layout.addWidget(QLabel("epochs:"), left_row, 0)
        param_layout.addWidget(self.spin_epochs, left_row, 1)
        left_row += 1

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 256)
        self.spin_batch.setValue(32)
        param_layout.addWidget(QLabel("batch_size:"), left_row, 0)
        param_layout.addWidget(self.spin_batch, left_row, 1)
        left_row += 1

        self.spin_emb = QSpinBox()
        self.spin_emb.setRange(8, 1024)
        self.spin_emb.setValue(128)
        param_layout.addWidget(QLabel("embedding_dim:"), left_row, 0)
        param_layout.addWidget(self.spin_emb, left_row, 1)
        left_row += 1

        self.spin_hidden = QSpinBox()
        self.spin_hidden.setRange(8, 2048)
        self.spin_hidden.setValue(256)
        param_layout.addWidget(QLabel("hidden_dim:"), left_row, 0)
        param_layout.addWidget(self.spin_hidden, left_row, 1)
        left_row += 1

        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(1e-6, 1.0)
        self.spin_lr.setDecimals(6)
        self.spin_lr.setValue(0.001)
        param_layout.addWidget(QLabel("lr:"), left_row, 0)
        param_layout.addWidget(self.spin_lr, left_row, 1)
        left_row += 1

        self.spin_wd = QDoubleSpinBox()
        self.spin_wd.setRange(0.0, 1.0)
        self.spin_wd.setDecimals(8)
        self.spin_wd.setValue(1e-5)
        param_layout.addWidget(QLabel("weight_decay:"), left_row, 0)
        param_layout.addWidget(self.spin_wd, left_row, 1)
        left_row += 1

        self.spin_min_char = QSpinBox()
        self.spin_min_char.setRange(1, 10)
        self.spin_min_char.setValue(1)
        param_layout.addWidget(QLabel("min_char_freq:"), left_row, 0)
        param_layout.addWidget(self.spin_min_char, left_row, 1)
        left_row += 1

        self.spin_max_samples = QSpinBox()
        self.spin_max_samples.setRange(0, 500000000)
        self.spin_max_samples.setValue(0)
        param_layout.addWidget(QLabel("max_samples(0=全量):"), left_row, 0)
        param_layout.addWidget(self.spin_max_samples, left_row, 1)
        left_row += 1

        self.chk_pin_memory = QPushButton("pin_memory")  # 用按钮样式做开关
        self.chk_pin_memory.setCheckable(True)
        self.chk_pin_memory.setChecked(True)
        self.chk_use_amp = QPushButton("use_amp")
        self.chk_use_amp.setCheckable(True)
        self.chk_use_amp.setChecked(True)
        param_layout.addWidget(self.chk_pin_memory, left_row, 0)
        param_layout.addWidget(self.chk_use_amp, left_row, 1)
        left_row += 1

        self.spin_num_workers = QSpinBox()
        self.spin_num_workers.setRange(0, 16)
        self.spin_num_workers.setValue(4)
        param_layout.addWidget(QLabel("num_workers:"), right_row, 2)
        param_layout.addWidget(self.spin_num_workers, right_row, 3)
        right_row += 1

        self.spin_prefetch_factor = QSpinBox()
        self.spin_prefetch_factor.setRange(1, 16)
        self.spin_prefetch_factor.setValue(2)
        param_layout.addWidget(QLabel("prefetch_factor:"), right_row, 2)
        param_layout.addWidget(self.spin_prefetch_factor, right_row, 3)
        right_row += 1

        self.spin_omp = QSpinBox()
        self.spin_omp.setRange(1, 16)
        self.spin_omp.setValue(1)
        param_layout.addWidget(QLabel("omp_threads:"), right_row, 2)
        param_layout.addWidget(self.spin_omp, right_row, 3)
        right_row += 1

        self.spin_clip_grad_norm = QDoubleSpinBox()
        self.spin_clip_grad_norm.setRange(0.1, 100.0)
        self.spin_clip_grad_norm.setDecimals(2)
        self.spin_clip_grad_norm.setValue(5.0)
        param_layout.addWidget(QLabel("clip_grad_norm:"), right_row, 2)
        param_layout.addWidget(self.spin_clip_grad_norm, right_row, 3)
        right_row += 1

        layout.addWidget(param_group)
        param_group.setLayout(param_layout)

        self._apply_preset_by_name("GPU 平衡（推荐）")
        self.cmb_preset.currentTextChanged.connect(self._on_preset_changed)

        # ---------- 底部双栏容器 ----------
        bottom_layout = QHBoxLayout()
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()

        # ---------- 左栏：语料选择 ----------
        corpus_group = QGroupBox("训练语料选择（单选 / 多选）")
        corpus_layout = QHBoxLayout()

        self.rb_single = QRadioButton("单选")
        self.rb_single.setChecked(True)
        self.rb_multi = QRadioButton("多选")

        mode_box = QButtonGroup(self)
        mode_box.addButton(self.rb_single)
        mode_box.addButton(self.rb_multi)

        left = QVBoxLayout()
        left.addWidget(QLabel("模式："))
        left.addWidget(self.rb_single)
        left.addWidget(self.rb_multi)

        self.combo_single = QComboBox()
        for stem in sorted(self._corpus_files.keys()):
            self.combo_single.addItem(stem)

        self.list_multi = QListWidget()
        self.list_multi.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for stem in sorted(self._corpus_files.keys()):
            item = QListWidgetItem(stem)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.list_multi.addItem(item)

        left2 = QVBoxLayout()
        left2.addWidget(QLabel("单选语料："))
        left2.addWidget(self.combo_single)
        left2.addWidget(QLabel("多选语料："))
        left2.addWidget(self.list_multi)

        self.rb_single.toggled.connect(self._on_corpus_mode_changed)
        self._on_corpus_mode_changed(self.rb_single.isChecked())

        corpus_layout.addLayout(left)
        corpus_layout.addLayout(left2)
        corpus_group.setLayout(corpus_layout)
        left_col.addWidget(corpus_group)

        # ---------- 输出目录 ----------
        out_group = QGroupBox("输出目录（每个语料会生成子目录）")
        out_layout = QHBoxLayout()
        self.txt_output_dir = QComboBox()
        self.txt_output_dir.setEditable(True)
        self.txt_output_dir.addItem(self.default_output_dir)
        out_layout.addWidget(self.txt_output_dir, 1)
        btn_pick_out = QPushButton("选择目录...")
        out_layout.addWidget(btn_pick_out)
        left_col.addWidget(out_group)
        out_group.setLayout(out_layout)

        btn_pick_out.clicked.connect(self._pick_output_dir)

        # ---------- 左栏：操作 ----------
        btn_group = QGroupBox("操作")
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶ 训练 + 自动评估（所选语料）")
        self.btn_start.setMinimumHeight(44)
        self.btn_clear_log = QPushButton("清空日志")
        self.btn_clear_log.setMinimumHeight(44)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_clear_log)
        btn_group.setLayout(btn_layout)
        left_col.addWidget(btn_group)

        # ---------- 左栏：评估结果 ----------
        result_group = QGroupBox("评估结果（P / R / F1）")
        result_layout = QVBoxLayout()
        self.tbl = QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(["语料", "Precision", "Recall", "F1"])
        result_layout.addWidget(self.tbl)
        result_group.setLayout(result_layout)
        left_col.addWidget(result_group)
        left_col.addStretch(1)

        # ---------- 右栏：运行日志 ----------
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Microsoft YaHei", 9))
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        right_col.addWidget(log_group, stretch=3)

        # ---------- 右栏：训练进度 ----------
        progress_group = QGroupBox("训练进度")
        progress_layout = QVBoxLayout()
        progress_layout.addWidget(QLabel("按当前 epoch/句子估算，仅用于参考"))
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        progress_layout.addWidget(self.progress)
        progress_group.setLayout(progress_layout)
        right_col.addWidget(progress_group, stretch=1)

        # ---------- 右栏：训练损失曲线 ----------
        loss_group = QGroupBox("训练损失曲线（train_history.json）")
        loss_layout = QVBoxLayout()
        self.loss_tbl = QTableWidget(0, 2)
        self.loss_tbl.setHorizontalHeaderLabels(["Epoch", "AvgSentenceLoss"])
        self.loss_tbl.horizontalHeader().setStretchLastSection(True)
        loss_layout.addWidget(self.loss_tbl)
        loss_group.setLayout(loss_layout)
        right_col.addWidget(loss_group, stretch=2)

        bottom_layout.addLayout(left_col, 1)
        bottom_layout.addLayout(right_col, 1)
        layout.addLayout(bottom_layout, 1)

        self.btn_start.clicked.connect(self._on_start_clicked)
        self.btn_clear_log.clicked.connect(lambda: self.log_text.clear())

    def _on_corpus_mode_changed(self, is_single: bool) -> None:
        self.combo_single.setEnabled(is_single)
        self.list_multi.setEnabled(not is_single)

    def _on_preset_changed(self, name: str) -> None:
        self._apply_preset_by_name(name)

    def _apply_preset_by_name(self, name: str) -> None:
        presets: Dict[str, Dict[str, object]] = {
            "Debug CPU（跑通流程）": {
                "device": "cpu",
                "epochs": 1,
                "batch_size": 8,
                "embedding_dim": 32,
                "hidden_dim": 64,
                "lr": 0.001,
                "weight_decay": 1e-5,
                "min_char_freq": 1,
                "max_samples": 1000,
                "pin_memory": False,
                "use_amp": False,
                "num_workers": 0,
                "prefetch_factor": 2,
                "omp_threads": 1,
                "clip_grad_norm": 5.0,
            },
            "GPU 平衡（推荐）": {
                "device": "cuda",
                "epochs": 10,
                "batch_size": 32,
                "embedding_dim": 128,
                "hidden_dim": 256,
                "lr": 0.001,
                "weight_decay": 1e-5,
                "min_char_freq": 1,
                "max_samples": 0,
                "pin_memory": True,
                "use_amp": True,
                "num_workers": 4,
                "prefetch_factor": 2,
                "omp_threads": 1,
                "clip_grad_norm": 5.0,
            },
            "GPU 快速（更快但更稳）": {
                "device": "cuda",
                "epochs": 6,
                "batch_size": 48,
                "embedding_dim": 128,
                "hidden_dim": 256,
                "lr": 0.001,
                "weight_decay": 1e-5,
                "min_char_freq": 1,
                "max_samples": 0,
                "pin_memory": True,
                "use_amp": True,
                "num_workers": 6,
                "prefetch_factor": 2,
                "omp_threads": 1,
                "clip_grad_norm": 5.0,
            },
            "GPU 保守（防 OOM）": {
                "device": "cuda",
                "epochs": 10,
                "batch_size": 24,
                "embedding_dim": 96,
                "hidden_dim": 192,
                "lr": 0.001,
                "weight_decay": 1e-5,
                "min_char_freq": 1,
                "max_samples": 0,
                "pin_memory": True,
                "use_amp": True,
                "num_workers": 2,
                "prefetch_factor": 2,
                "omp_threads": 1,
                "clip_grad_norm": 5.0,
            },
        }

        cfg = presets.get(name)
        if not cfg:
            return

        self.cmb_device.setCurrentText(str(cfg["device"]))
        self.spin_epochs.setValue(int(cfg["epochs"]))
        self.spin_batch.setValue(int(cfg["batch_size"]))
        self.spin_emb.setValue(int(cfg["embedding_dim"]))
        self.spin_hidden.setValue(int(cfg["hidden_dim"]))
        self.spin_lr.setValue(float(cfg["lr"]))
        self.spin_wd.setValue(float(cfg["weight_decay"]))
        self.spin_min_char.setValue(int(cfg["min_char_freq"]))
        self.spin_max_samples.setValue(int(cfg["max_samples"]))

        self.chk_pin_memory.setChecked(bool(cfg["pin_memory"]))
        self.chk_use_amp.setChecked(bool(cfg["use_amp"]))

        self.spin_num_workers.setValue(int(cfg["num_workers"]))
        self.spin_prefetch_factor.setValue(int(cfg["prefetch_factor"]))
        self.spin_omp.setValue(int(cfg["omp_threads"]))
        self.spin_clip_grad_norm.setValue(float(cfg["clip_grad_norm"]))

    def _append_log(self, text: str) -> None:
        if not text:
            return
        # tqdm 会不断刷新同一行，直接 append 会很乱；这里做“尽量追加 + 去掉尾部空白”
        for line in text.splitlines():
            line = line.strip("\r\n")
            if not line:
                continue
            self.log_text.append(line)

    def _set_running(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_clear_log.setEnabled(not running)

    def _get_selected_stems(self) -> List[str]:
        if self.rb_single.isChecked():
            return [self.combo_single.currentText()]

        stems = []
        for i in range(self.list_multi.count()):
            item = self.list_multi.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                stems.append(item.text())
        return stems

    def _make_train_cmd(self, stem: str, out_dir: str) -> List[str]:
        train_path = self._corpus_files[stem]
        cmd = [
            sys.executable,
            self.main_py,
            "--mode",
            "train",
            "--train_dir",
            train_path,
            "--output_dir",
            out_dir,
            "--epochs",
            str(self.spin_epochs.value()),
            "--batch_size",
            str(self.spin_batch.value()),
            "--embedding_dim",
            str(self.spin_emb.value()),
            "--hidden_dim",
            str(self.spin_hidden.value()),
            "--lr",
            str(self.spin_lr.value()),
            "--weight_decay",
            str(self.spin_wd.value()),
            "--min_char_freq",
            str(self.spin_min_char.value()),
            "--max_samples",
            str(self.spin_max_samples.value()),
            "--device",
            self.cmb_device.currentText(),
            "--num_workers",
            str(self.spin_num_workers.value()),
            "--prefetch_factor",
            str(self.spin_prefetch_factor.value()),
            "--clip_grad_norm",
            str(self.spin_clip_grad_norm.value()),
            "--omp_threads",
            str(self.spin_omp.value()),
        ]

        if self.chk_pin_memory.isChecked():
            cmd.append("--pin_memory")
        if self.chk_use_amp.isChecked():
            cmd.append("--use_amp")

        return cmd

    def _make_eval_cmd(self, stem: str, out_dir: str) -> List[str]:
        model_path = os.path.join(out_dir, "bilstm_crf.pt")
        vocab_path = os.path.join(out_dir, "char_vocab.json")
        cmd = [
            sys.executable,
            self.main_py,
            "--mode",
            "eval",
            "--test_dir",
            self.test_dir,
            "--model_path",
            model_path,
            "--vocab_path",
            vocab_path,
            "--embedding_dim",
            str(self.spin_emb.value()),
            "--hidden_dim",
            str(self.spin_hidden.value()),
            "--device",
            self.cmb_device.currentText(),
        ]
        return cmd

    def _progress_from_line(self, line: str) -> None:
        # 样式示例：
        # Epoch 1/10:   2%|▍ | 816/54247 [05:06<5:11:52, 2.86batch/s, avg_loss=9.6135]
        m = re.search(r"Epoch\s+(\d+)/(\d+):.*?(\d+)/(\d+)", line)
        if not m:
            return
        epoch_cur = int(m.group(1))
        epoch_total = int(m.group(2))
        batch_cur = int(m.group(3))
        batch_total = int(m.group(4))
        if epoch_total <= 0 or batch_total <= 0:
            return
        ratio_epoch = (epoch_cur - 1 + batch_cur / batch_total) / epoch_total
        self.progress.setValue(max(0, min(100, int(ratio_epoch * 100))))

    def _start_subprocess(self, cmd: List[str]) -> None:
        if self.process:
            try:
                self.process.kill()
            except Exception:
                pass
        self.process = QProcess(self)
        self.process.setWorkingDirectory(self._work_dir)
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.readyReadStandardError.connect(self._on_stderr)
        self.process.finished.connect(self._on_finished)

        self._append_log(f"[CMD] {' '.join(cmd)}")
        self.process.start(cmd[0], cmd[1:])

    def _on_stdout(self) -> None:
        if not self.process:
            return
        data = self.process.readAllStandardOutput().data()
        text = data.decode("utf-8", errors="replace")
        if text:
            self._append_log(text)
            last_line = text.splitlines()[-1] if text.splitlines() else ""
            self._progress_from_line(last_line)

    def _on_stderr(self) -> None:
        if not self.process:
            return
        data = self.process.readAllStandardError().data()
        text = data.decode("utf-8", errors="replace")
        if text:
            self._append_log(text)

    def _on_finished(self, code: int, _status) -> None:
        if not self.current_corpus_stem:
            return
        stem = self.current_corpus_stem
        if code != 0:
            self._set_running(False)
            QMessageBox.critical(self, "执行失败", f"{stem} 执行失败，exit code={code}")
            return

        # 根据 stage 切换：train -> eval
        if self.stage == "train":
            # eval
            out_dir = os.path.join(self._get_output_dir(), stem)
            self._update_loss_table_from_history(out_dir)
            self.stage = "eval"
            self._start_subprocess(self._make_eval_cmd(stem, out_dir))
            return

        if self.stage == "eval":
            # 从输出 dir 中读取 metrics（eval 返回 stdout 的 json）
            out_dir = os.path.join(self._get_output_dir(), stem)
            model_path = os.path.join(out_dir, "bilstm_crf.pt")
            vocab_path = os.path.join(out_dir, "char_vocab.json")
            if os.path.exists(model_path) and os.path.exists(vocab_path):
                pass

            # 为了不复杂解析 stdout，这里直接重跑一次 eval 的结果解析：
            # 直接调用 evaluate 返回字典会阻塞 GUI，所以我们只从最近日志里解析 json。
            # 解析失败则留空，且不影响后续语料。
            metrics = self._parse_last_metrics_from_log(stem)
            if metrics:
                self.metrics_by_stem[stem] = metrics
                self._update_results_table(stem, metrics)

            self.stage = "done"
            self._run_next_corpus()
            return

    def _update_loss_table_from_history(self, out_dir: str) -> None:
        history_path = os.path.join(out_dir, "train_history.json")
        if not os.path.exists(history_path):
            return
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            history = data.get("history", [])
        except Exception:
            return

        self.loss_tbl.setRowCount(0)
        for item in history:
            epoch = item.get("epoch", 0)
            loss = item.get("avg_sentence_loss", 0.0)
            r = self.loss_tbl.rowCount()
            self.loss_tbl.insertRow(r)
            self.loss_tbl.setItem(r, 0, QTableWidgetItem(str(epoch)))
            self.loss_tbl.setItem(r, 1, QTableWidgetItem(f"{loss:.6f}"))

    def _parse_last_metrics_from_log(self, stem: str) -> Optional[Dict[str, float]]:
        # 从日志最后一段中找 { "precision": ..., "recall": ..., "f1": ... }
        # 这里做宽松匹配；成功率比严格解析高。
        content = self.log_text.toPlainText()
        # 精简匹配
        matches = re.findall(
            r'"precision"\s*:\s*([0-9.]+).*?"recall"\s*:\s*([0-9.]+).*?"f1"\s*:\s*([0-9.]+)',
            content,
        )
        if not matches:
            return None
        p, r, f = matches[-1]
        return {"precision": _safe_float(p), "recall": _safe_float(r), "f1": _safe_float(f)}

    def _update_results_table(self, stem: str, metrics: Dict[str, float]) -> None:
        # 查找是否已有行
        for r in range(self.tbl.rowCount()):
            if self.tbl.item(r, 0) and self.tbl.item(r, 0).text() == stem:
                self.tbl.setItem(r, 1, QTableWidgetItem(f"{metrics['precision']:.6f}"))
                self.tbl.setItem(r, 2, QTableWidgetItem(f"{metrics['recall']:.6f}"))
                self.tbl.setItem(r, 3, QTableWidgetItem(f"{metrics['f1']:.6f}"))
                return

        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        self.tbl.setItem(r, 0, QTableWidgetItem(stem))
        self.tbl.setItem(r, 1, QTableWidgetItem(f"{metrics['precision']:.6f}"))
        self.tbl.setItem(r, 2, QTableWidgetItem(f"{metrics['recall']:.6f}"))
        self.tbl.setItem(r, 3, QTableWidgetItem(f"{metrics['f1']:.6f}"))

    def _on_start_clicked(self) -> None:
        stems = self._get_selected_stems()
        if not stems:
            QMessageBox.warning(self, "提示", "请选择至少一个语料文件。")
            return

        self.metrics_by_stem.clear()
        self.tbl.setRowCount(0)
        self.log_text.clear()
        self.progress.setValue(0)

        self._set_running(True)
        self.stems_queue = stems
        self._run_next_corpus()

    def _run_next_corpus(self) -> None:
        if not hasattr(self, "stems_queue") or not self.stems_queue:
            self._set_running(False)
            self.stage = None
            self.current_corpus_stem = None
            self.progress.setValue(100)
            QMessageBox.information(self, "完成", "训练 + 评估全部完成。")
            return

        self.current_corpus_stem = self.stems_queue.pop(0)
        stem = self.current_corpus_stem
        out_dir = os.path.join(self._get_output_dir(), stem)
        self.stage = "train"
        self._start_subprocess(self._make_train_cmd(stem, out_dir))

    def _get_output_dir(self) -> str:
        text = self.txt_output_dir.currentText().strip()
        return text if text else self.default_output_dir

    def _pick_output_dir(self) -> None:
        start = self._get_output_dir()
        chosen = QFileDialog.getExistingDirectory(self, "选择输出目录", start)
        if chosen:
            # QComboBox is editable; set text via lineEdit
            self.txt_output_dir.setEditText(chosen)


def main() -> None:
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))
    w = NNCRFTrainEvalWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

