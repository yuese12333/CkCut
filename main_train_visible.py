#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CkCut 训练流水线可视化：通过按钮执行 A/B/C/D/E 步骤，并支持调参。
"""

import os
import sys

# 抑制 Qt 字体/DirectWrite 相关警告（不影响运行，仅减少终端刷屏）
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false;qt.text.font.db=false")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox,
    QTextEdit, QLabel, QScrollArea, QMessageBox, QDialog, QLineEdit,
    QGridLayout, QFrame,
)
from PyQt6.QtCore import QProcess, QTimer, Qt
from PyQt6.QtGui import QFont


def _get_base_dir():
    if getattr(sys, "frozen", False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = _get_base_dir()

# 与 main.py 一致的路径（运行态用项目根目录，打包后若需可改为 _get_base_dir）
if getattr(sys, "frozen", False):
    _work_dir = os.path.dirname(sys.executable)
else:
    _work_dir = os.path.dirname(os.path.abspath(__file__))

RAW_CORPUS_DIR = os.path.join(_work_dir, "data", "raw_corpus")
PROCESSED_FILE = os.path.join(_work_dir, "data", "processed", "merged_cleaned.txt")
OUTPUT_DICT = os.path.join(_work_dir, "data", "output_dict", "my_dict_wiki.txt")
HMM_TRAIN_DIR = os.path.join(_work_dir, "data", "HMM_train")
HMM_MODEL = os.path.join(_work_dir, "data", "output_dict", "hmm_model.json")
TEST_DIR = os.path.join(_work_dir, "data", "evaluate")
MAIN_PY = os.path.join(_work_dir, "main.py")


class SegmentDialog(QDialog):
    """步骤 E：交互式分词小窗口"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔪 步骤 E - 交互式分词")
        self.resize(520, 320)
        layout = QVBoxLayout(self)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("输入中文句子，回车得到分词结果...")
        self.input_edit.returnPressed.connect(self.do_segment)
        self.result_label = QLabel("分词结果将显示在这里")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("background: #f5f5f5; padding: 8px; border-radius: 4px;")
        layout.addWidget(QLabel("输入句子:"))
        layout.addWidget(self.input_edit)
        layout.addWidget(QLabel("分词结果:"))
        layout.addWidget(self.result_label)
        self.segmenter = None

    def set_segmenter(self, segmenter):
        self.segmenter = segmenter

    def do_segment(self):
        if not self.segmenter:
            self.result_label.setText("未加载分词器，请先完成 B 步骤并确保词典存在。")
            return
        text = self.input_edit.text().strip()
        if not text:
            return
        words = self.segmenter.cut(text)
        self.result_label.setText(" / ".join(words))


class TrainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚀 CkCut 训练流水线")
        self.resize(820, 620)
        self.process = None
        self.build_ui()

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # ---------- 超参数 ----------
        param_group = QGroupBox("超参数 (用于 B 步骤新词发现)")
        param_layout = QGridLayout()
        param_layout.addWidget(QLabel("候选词最大长度 max_len:"), 0, 0)
        self.max_len = QSpinBox()
        self.max_len.setRange(2, 10)
        self.max_len.setValue(4)
        param_layout.addWidget(self.max_len, 0, 1)
        param_layout.addWidget(QLabel("最低词频 min_freq:"), 1, 0)
        self.min_freq = QSpinBox()
        self.min_freq.setRange(1, 100)
        self.min_freq.setValue(5)
        param_layout.addWidget(self.min_freq, 1, 1)
        param_layout.addWidget(QLabel("最低 PMI min_pmi:"), 2, 0)
        self.min_pmi = QDoubleSpinBox()
        self.min_pmi.setRange(0.5, 20.0)
        self.min_pmi.setValue(4.0)
        self.min_pmi.setDecimals(1)
        param_layout.addWidget(self.min_pmi, 2, 1)
        param_layout.addWidget(QLabel("最低边界熵 min_entropy:"), 3, 0)
        self.min_entropy = QDoubleSpinBox()
        self.min_entropy.setRange(0.0, 5.0)
        self.min_entropy.setValue(1.0)
        self.min_entropy.setDecimals(1)
        param_layout.addWidget(self.min_entropy, 3, 1)
        self.withouthmm = QCheckBox("评测/分词时禁用 HMM (纯 DAG)")
        param_layout.addWidget(self.withouthmm, 4, 0, 1, 2)
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # ---------- 步骤按钮 ----------
        btn_group = QGroupBox("流水线步骤 (点击执行)")
        btn_layout = QHBoxLayout()
        self.btn_a = QPushButton("🛠️ A\n预处理")
        self.btn_b = QPushButton("🧠 B\n新词发现")
        self.btn_c = QPushButton("🤖 C\nHMM 训练")
        self.btn_d = QPushButton("📈 D\n评测")
        self.btn_e = QPushButton("🔪 E\n交互分词")
        self.btn_all = QPushButton("▶ A→B→C→D\n一键顺序")
        for b in (self.btn_a, self.btn_b, self.btn_c, self.btn_d, self.btn_e, self.btn_all):
            b.setMinimumHeight(52)
            b.clicked.connect(self._make_runner(b))
        btn_layout.addWidget(self.btn_a)
        btn_layout.addWidget(self.btn_b)
        btn_layout.addWidget(self.btn_c)
        btn_layout.addWidget(self.btn_d)
        btn_layout.addWidget(self.btn_e)
        btn_layout.addWidget(self.btn_all)
        btn_group.setLayout(btn_layout)
        layout.addWidget(btn_group)

        # ---------- 日志 ----------
        log_group = QGroupBox("运行日志")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Microsoft YaHei", 9))
        self.log_text.setPlaceholderText("执行步骤后，标准输出会显示在这里...")
        log_layout = QVBoxLayout()
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

    def _make_runner(self, btn):
        def run():
            if btn == self.btn_a:
                self.run_step("A")
            elif btn == self.btn_b:
                self.run_step("B")
            elif btn == self.btn_c:
                self.run_step("C")
            elif btn == self.btn_d:
                self.run_step("D")
            elif btn == self.btn_e:
                self.run_step_e_in_gui()
            elif btn == self.btn_all:
                self.run_step("ABCD")
        return run

    def run_step_e_in_gui(self):
        """步骤 E：在 GUI 内加载分词器并打开交互对话框"""
        if not os.path.exists(OUTPUT_DICT):
            QMessageBox.warning(self, "提示", f"未找到词典文件，请先执行 B 步骤。\n{OUTPUT_DICT}")
            return
        try:
            from src.segmenter import AutoSegmenter
            use_hmm = None if self.withouthmm.isChecked() else HMM_MODEL
            seg = AutoSegmenter(OUTPUT_DICT, use_hmm)
            dlg = SegmentDialog(self)
            dlg.set_segmenter(seg)
            dlg.exec()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载分词器失败: {e}")

    def run_step(self, step: str):
        if self.process and self.process.state() == QProcess.ProcessState.Running:
            self.log_text.append("\n⚠️ 已有任务在运行，请等待完成。")
            return
        if not os.path.exists(MAIN_PY):
            self.log_text.append(f"\n❌ 未找到 main.py: {MAIN_PY}")
            return
        self.log_text.append(f"\n{'='*50}\n▶ 执行步骤: {step}\n{'='*50}")
        self.set_buttons_enabled(False)
        self.process = QProcess(self)
        self.process.setWorkingDirectory(_work_dir)
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.readyReadStandardError.connect(self._on_stderr)
        self.process.finished.connect(self._on_finished)
        args = [
            sys.executable, MAIN_PY,
            "--step", step,
            "--max_len", str(self.max_len.value()),
            "--min_freq", str(self.min_freq.value()),
            "--min_pmi", str(self.min_pmi.value()),
            "--min_entropy", str(self.min_entropy.value()),
        ]
        if self.withouthmm.isChecked():
            args.append("--withouthmm")
        self.process.start(args[0], args[1:])

    def _on_stdout(self):
        if self.process:
            data = self.process.readAllStandardOutput().data().decode("utf-8", errors="replace")
            self.log_text.append(data.rstrip())

    def _on_stderr(self):
        if self.process:
            data = self.process.readAllStandardError().data().decode("utf-8", errors="replace")
            self.log_text.append(f"[stderr] {data.rstrip()}")

    def _on_finished(self, code, status):
        self.set_buttons_enabled(True)
        self.log_text.append(f"\n✅ 步骤结束 (exit code: {code})\n")

    def set_buttons_enabled(self, enabled: bool):
        for b in (self.btn_a, self.btn_b, self.btn_c, self.btn_d, self.btn_e, self.btn_all):
            b.setEnabled(enabled)


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))
    w = TrainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
