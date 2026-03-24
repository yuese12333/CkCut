import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QTextEdit, QPushButton, QLabel,
                            QFileDialog, QMessageBox, QStatusBar, QFrame,
                            QCheckBox, QComboBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

# 导入我们的分词引擎
from src.segmenter import AutoSegmenter
from src_nn_crf.infer import CRFSegmenter
import json

class CkCutWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # DAG 词典二选一（与 CRF 无关）；HMM 模型路径固定
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        _od = os.path.join(self.base_dir, "data", "output_dict")
        self.DICT_PRESETS = [
            ("Wiki 大词典", os.path.join(_od, "my_dict_wiki.txt")),
            ("双小说训练词典", os.path.join(_od, "my_dict_primary.txt")),
        ]
        self.dict_path = self.DICT_PRESETS[0][1]
        self.hmm_path = os.path.join(_od, "hmm_model.json")
        # CRF 预置：显示名 -> bilstm_crf.pt 相对项目根路径（char_vocab.json 同目录）
        self.CRF_PRESETS = [
            ("合并训练 (aggressive)", "data/output_nn_crf_merged_gpu_aggressive/bilstm_crf.pt"),
            ("单语料: as_train", "data/output_nn_crf_single/as_train/bilstm_crf.pt"),
            ("单语料: cityu_train", "data/output_nn_crf_single/cityu_train/bilstm_crf.pt"),
            ("单语料: msr_training", "data/output_nn_crf_single/msr_training/bilstm_crf.pt"),
            ("单语料: pku_training", "data/output_nn_crf_single/pku_training/bilstm_crf.pt"),
        ]
        _default_pt = os.path.join(self.base_dir, self.CRF_PRESETS[0][1])
        self.crf_model_path = _default_pt
        self.crf_vocab_path = os.path.join(os.path.dirname(_default_pt), "char_vocab.json")
        self.segmenter = None  # DAG/HMM 引擎
        self.crf_segmenter = None  # CRF 引擎
        
        self.init_ui()

    def init_ui(self):
        """初始化 UI 布局和控件"""
        self.setWindowTitle("🚀 CkCut 智能分词体验终端 v2.2")
        self.resize(950, 650)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # ================= 上方：双排控制台 =================
        
        # --- 第一排：固定资源说明（词典与 HMM 由默认路径加载，不可在界面中更换）---
        dict_layout = QHBoxLayout()
        self.lbl_current_dict = QLabel("当前资源: 未加载")
        self.lbl_current_dict.setStyleSheet("color: #555; font-style: italic;")
        dict_layout.addWidget(self.lbl_current_dict)
        dict_layout.addStretch()
        
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        
        # --- 第二排：文本与分词操作 ---
        text_op_layout = QHBoxLayout()
        self.btn_import = QPushButton("📂 智能导入 TXT")
        self.cmb_engine = QComboBox()
        self.cmb_engine.addItems(["DAG+HMM", "CRF"])
        self.cmb_engine.setCurrentText("DAG+HMM")
        self.cmb_engine.setToolTip("选择分词引擎模式：传统词典/HMM 或神经网络 CRF")

        self.lbl_dict = QLabel("DAG 词典:")
        self.cmb_dict = QComboBox()
        for label, _ in self.DICT_PRESETS:
            self.cmb_dict.addItem(label)
        self.cmb_dict.setCurrentIndex(0)
        self.cmb_dict.setToolTip(
            "DAG+HMM 使用的词典二选一：Wiki 大词典，或由两部小说语料训练得到的 my_dict_primary"
        )

        self.lbl_crf_model = QLabel("CRF 模型:")
        self.cmb_crf_model = QComboBox()
        for label, _ in self.CRF_PRESETS:
            self.cmb_crf_model.addItem(label)
        self.cmb_crf_model.setCurrentIndex(0)
        self.cmb_crf_model.setToolTip("预置 BiLSTM-CRF 权重与词表")
        self.lbl_crf_model.setVisible(False)
        self.cmb_crf_model.setVisible(False)
        
        # HMM 兜底开关
        self.chk_hmm = QCheckBox("✨ 启用 HMM 兜底 (识别未登录词)")
        self.chk_hmm.setChecked(True)
        self.chk_hmm.setToolTip("开启后将使用 Viterbi 算法智能缝合未被收录的陌生词汇。")
        self.chk_hmm.setStyleSheet("font-weight: bold; color: #d35400;")
        
        self.btn_segment = QPushButton("🔪 界面一键分词")
        self.btn_export = QPushButton("💾 导出界面结果")
        
        self.btn_segment.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        
        text_op_layout.addWidget(self.btn_import)
        text_op_layout.addWidget(QLabel("引擎模式:"))
        text_op_layout.addWidget(self.cmb_engine)
        text_op_layout.addWidget(self.lbl_dict)
        text_op_layout.addWidget(self.cmb_dict)
        text_op_layout.addWidget(self.lbl_crf_model)
        text_op_layout.addWidget(self.cmb_crf_model)
        text_op_layout.addWidget(self.chk_hmm)
        text_op_layout.addWidget(self.btn_segment)
        text_op_layout.addWidget(self.btn_export)
        
        main_layout.addLayout(dict_layout)
        main_layout.addWidget(line)
        main_layout.addLayout(text_op_layout)
        
        # ================= 下方：双栏文本区 =================
        text_layout = QHBoxLayout()
        
        left_layout = QVBoxLayout()
        left_label = QLabel("📝 原始文本:")
        left_label.setFont(QFont("Microsoft YaHei", 10, QFont.Weight.Bold))
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("请输入需要分词的中文句子，或者点击上方按钮导入 txt 文件...\n\n(大文件建议选择直接后台导出)")
        self.text_input.setFont(QFont("Microsoft YaHei", 11))
        left_layout.addWidget(left_label)
        left_layout.addWidget(self.text_input)
        
        right_layout = QVBoxLayout()
        right_label = QLabel("✂️ 分词结果:")
        right_label.setFont(QFont("Microsoft YaHei", 10, QFont.Weight.Bold))
        self.text_output = QTextEdit()
        self.text_output.setPlaceholderText("分词结果将在这里显示...")
        self.text_output.setFont(QFont("Microsoft YaHei", 11))
        self.text_output.setReadOnly(True)
        self.text_output.setStyleSheet("background-color: #f5f5f5;")
        right_layout.addWidget(right_label)
        right_layout.addWidget(self.text_output)
        
        text_layout.addLayout(left_layout)
        text_layout.addLayout(right_layout)
        
        main_layout.addLayout(text_layout)
        
        # ================= 状态栏 =================
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("初始化中...")

        # ================= 绑定事件 =================
        self.btn_import.clicked.connect(self.import_file)
        self.btn_segment.clicked.connect(self.run_segmentation)
        self.btn_export.clicked.connect(self.export_file)
        self.cmb_engine.currentTextChanged.connect(self._on_engine_changed)
        self.cmb_dict.currentIndexChanged.connect(self._on_dict_choice_changed)
        self.cmb_crf_model.currentIndexChanged.connect(self._on_crf_preset_changed)
        self._on_engine_changed(self.cmb_engine.currentText())

    # ---------------- 词典引擎逻辑 ----------------

    def _sync_top_resource_label(self):
        if self.cmb_engine.currentText() == "CRF":
            if self.crf_segmenter:
                self.lbl_current_dict.setText(
                    f"CRF: {self.cmb_crf_model.currentText()} | 待切 DAG 词典: {self._dict_status_label()}"
                )
                self.lbl_current_dict.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.lbl_current_dict.setText(
                    f"CRF 模型未加载 | 待切 DAG 词典: {self._dict_status_label()}"
                )
                self.lbl_current_dict.setStyleSheet("color: red; font-weight: bold;")
        else:
            if self.segmenter:
                self.lbl_current_dict.setText(f"DAG+HMM: {self._dict_status_label()}")
                self.lbl_current_dict.setStyleSheet("color: green; font-weight: bold;")
            elif os.path.exists(self.dict_path):
                self.lbl_current_dict.setText(f"DAG+HMM: {self._dict_status_label()}")
                self.lbl_current_dict.setStyleSheet("color: #555; font-style: italic;")
            else:
                self.lbl_current_dict.setText(f"缺失词典: {os.path.basename(self.dict_path)}")
                self.lbl_current_dict.setStyleSheet("color: red; font-weight: bold;")

    def _dag_segmenter_matches_dict(self) -> bool:
        if not self.segmenter:
            return False
        return os.path.normpath(self.segmenter.dict_path) == os.path.normpath(self.dict_path)

    def _on_dict_choice_changed(self, idx: int):
        if idx < 0 or idx >= len(self.DICT_PRESETS):
            return
        self.dict_path = self.DICT_PRESETS[idx][1]
        if self.cmb_engine.currentText() == "DAG+HMM":
            self.load_engine()
        else:
            self._sync_top_resource_label()

    def _apply_crf_preset_index(self, idx: int):
        """按预置项更新 CRF 路径并加载引擎。"""
        if idx < 0 or idx >= len(self.CRF_PRESETS):
            return
        _, rel_pt = self.CRF_PRESETS[idx]
        self.crf_model_path = os.path.join(self.base_dir, rel_pt)
        self.crf_vocab_path = os.path.join(os.path.dirname(self.crf_model_path), "char_vocab.json")
        self.load_crf_engine()
        if self.cmb_engine.currentText() == "CRF":
            self._sync_top_resource_label()

    def _on_crf_preset_changed(self, idx: int):
        if self.cmb_engine.currentText() != "CRF":
            return
        self._apply_crf_preset_index(idx)

    def _guess_crf_dims(self):
        """优先从同目录 train_history.json 推断 embedding/hidden 维度。"""
        default_dims = (128, 256)
        history_path = os.path.join(os.path.dirname(self.crf_model_path), "train_history.json")
        if not os.path.exists(history_path):
            return default_dims
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cfg = data.get("config", {})
            emb = int(cfg.get("embedding_dim", default_dims[0]))
            hid = int(cfg.get("hidden_dim", default_dims[1]))
            return emb, hid
        except Exception:
            return default_dims

    def _dict_status_label(self) -> str:
        return f"{self.cmb_dict.currentText()} ({os.path.basename(self.dict_path)})"

    def load_engine(self):
        """加载分词引擎：当前选中的单个词典 + HMM"""
        if not os.path.exists(self.dict_path):
            self.lbl_current_dict.setText(f"当前资源: 找不到词典 ({os.path.basename(self.dict_path)})")
            self.lbl_current_dict.setStyleSheet("color: red; font-weight: bold;")
            self.status_bar.showMessage("❌ 引擎加载失败：缺失词典文件")
            self.segmenter = None
            self.btn_segment.setEnabled(self.cmb_engine.currentText() == "CRF")
            self.chk_hmm.setEnabled(False)
            return

        self.status_bar.showMessage("⏳ 正在加载混合分词引擎，请稍候...")
        QApplication.processEvents()

        try:
            self.segmenter = AutoSegmenter(self.dict_path, self.hmm_path)

            if self.segmenter.hmm_enabled:
                self.chk_hmm.setEnabled(True)
                self.chk_hmm.setChecked(True)
                self.chk_hmm.setText("✨ 启用 HMM ")
            else:
                self.chk_hmm.setEnabled(False)
                self.chk_hmm.setChecked(False)
                self.chk_hmm.setText("⚠️ HMM 模型文件缺失")

            self._sync_top_resource_label()
            self.status_bar.showMessage(f"✅ 双引擎就绪！DAG 节点数: {len(self.segmenter.FREQ)}")
            self.btn_segment.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"词典加载异常: {str(e)}")
            self.status_bar.showMessage("❌ 引擎加载异常")
            self.segmenter = None
            self.btn_segment.setEnabled(self.cmb_engine.currentText() == "CRF")
            self.chk_hmm.setEnabled(False)

    def load_crf_engine(self):
        """加载 CRF 分词引擎。"""
        if not os.path.exists(self.crf_model_path) or not os.path.exists(self.crf_vocab_path):
            self.status_bar.showMessage("❌ CRF 引擎加载失败：缺失模型或词表")
            self.crf_segmenter = None
            return
        try:
            emb, hid = self._guess_crf_dims()
            self.crf_segmenter = CRFSegmenter(
                model_path=self.crf_model_path,
                vocab_path=self.crf_vocab_path,
                embedding_dim=emb,
                hidden_dim=hid,
                device="auto",
            )
            self.status_bar.showMessage(f"✅ CRF 引擎就绪！模型: {os.path.basename(self.crf_model_path)}")
        except Exception as e:
            self.crf_segmenter = None
            self.status_bar.showMessage(f"❌ CRF 引擎加载失败: {e}")

    def _on_engine_changed(self, mode: str):
        using_crf = (mode == "CRF")
        self.lbl_dict.setVisible(not using_crf)
        self.cmb_dict.setVisible(not using_crf)
        self.lbl_crf_model.setVisible(using_crf)
        self.cmb_crf_model.setVisible(using_crf)
        self.chk_hmm.setEnabled(not using_crf and self.segmenter is not None and self.segmenter.hmm_enabled)
        if using_crf:
            self._apply_crf_preset_index(self.cmb_crf_model.currentIndex())
            self.status_bar.showMessage("🔁 当前为 CRF 分词模式")
        else:
            idx = self.cmb_dict.currentIndex()
            if idx >= 0:
                self.dict_path = self.DICT_PRESETS[idx][1]
            if not self._dag_segmenter_matches_dict():
                self.load_engine()
            else:
                self.chk_hmm.setEnabled(self.segmenter is not None and self.segmenter.hmm_enabled)
                self._sync_top_resource_label()
            self.status_bar.showMessage("🔁 当前为 DAG+HMM 分词模式")

    # ---------------- 文本智能导入流转逻辑 ----------------

    def import_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文本文件", self.base_dir, "Text Files (*.txt);;All Files (*)")
        if not file_path:
            return
            
        try:
            # 预判文件大小 (500 KB 约等于 20-30 万汉字)
            file_size = os.path.getsize(file_path)
            LARGE_FILE_THRESHOLD = 500 * 1024
            
            if file_size > LARGE_FILE_THRESHOLD:
                QMessageBox.information(self, "文件体积过大",
                                        "该文件体积过大，为了防止界面卡死和内存溢出，将直接进入【后台分词并导出】模式。")
                self.process_file_directly(file_path)
            else:
                # 文件大小适中，让用户自己选择
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("导入选项")
                msg_box.setText("请选择如何处理该文本文件：")
                btn_gui = msg_box.addButton("显示在界面上预览", QMessageBox.ButtonRole.ActionRole)
                btn_export = msg_box.addButton("直接后台分词并导出", QMessageBox.ButtonRole.ActionRole)
                msg_box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
                
                msg_box.exec()
                
                if msg_box.clickedButton() == btn_gui:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        self.text_input.setPlainText(content)
                        self.status_bar.showMessage(f"📁 成功导入预览: {os.path.basename(file_path)}")
                elif msg_box.clickedButton() == btn_export:
                    self.process_file_directly(file_path)
                    
        except Exception as e:
            QMessageBox.warning(self, "操作失败", f"发生错误:\n{str(e)}")

    def process_file_directly(self, input_path):
        """流式处理大文件，边读边切边存，内存开销极低"""
        if self.cmb_engine.currentText() == "CRF":
            if not self.crf_segmenter:
                self.load_crf_engine()
            if not self.crf_segmenter:
                QMessageBox.warning(self, "提示", "CRF 引擎未加载成功，请检查模型与词表文件。")
                return
        elif not self.segmenter:
            QMessageBox.warning(self, "提示", "DAG+HMM 引擎未就绪，请确认 data/output_dict 下默认词典文件存在。")
            return
            
        save_path, _ = QFileDialog.getSaveFileName(self, "保存分词结果", self.base_dir, "Text Files (*.txt)")
        if not save_path:
            return
            
        self.status_bar.showMessage("🔪 正在后台流式分词中，请耐心等候...")
        # 冻结所有可能干扰后台进程的按钮
        self.btn_segment.setEnabled(False)
        self.btn_import.setEnabled(False)
        self.cmb_dict.setEnabled(False)
        self.chk_hmm.setEnabled(False)
        QApplication.processEvents()
        
        # 同步当前的 HMM 开关状态（CRF 模式不适用）
        if self.cmb_engine.currentText() == "DAG+HMM":
            self.segmenter.hmm_enabled = self.chk_hmm.isChecked()
        
        try:
            line_count = 0
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as fin, \
                open(save_path, 'w', encoding='utf-8') as fout:
                
                for line in fin:
                    line = line.strip()
                    if not line:
                        fout.write("\n")
                        continue
                        
                    if self.cmb_engine.currentText() == "CRF":
                        words = self.crf_segmenter.cut(line)
                    else:
                        words = self.segmenter.cut(line)
                    fout.write(" / ".join(words) + "\n")
                    
                    line_count += 1
                    # 防假死心跳：每处理 500 行，刷新一次 GUI 消息队列
                    if line_count % 500 == 0:
                        self.status_bar.showMessage(f"🔪 后台流式分词中... 已处理 {line_count} 行")
                        QApplication.processEvents()
                        
            self.status_bar.showMessage(f"✨ 后台分词完成！共处理 {line_count} 行。")
            QMessageBox.information(self, "成功", f"大文件分词完成并导出至:\n{os.path.basename(save_path)}")
        except Exception as e:
            QMessageBox.critical(self, "分词错误", f"文件处理过程中发生异常:\n{str(e)}")
            self.status_bar.showMessage("❌ 后台分词失败")
        finally:
            # 恢复按钮状态
            self.btn_segment.setEnabled(True)
            self.btn_import.setEnabled(True)
            crf_on = self.cmb_engine.currentText() == "CRF"
            self.cmb_dict.setEnabled(not crf_on)
            self.chk_hmm.setEnabled(
                not crf_on and self.segmenter is not None and self.segmenter.hmm_enabled
            )

    # ---------------- 界面文本分词逻辑 ----------------

    def run_segmentation(self):
        if self.cmb_engine.currentText() == "CRF":
            if not self.crf_segmenter:
                self.load_crf_engine()
            if not self.crf_segmenter:
                QMessageBox.warning(self, "提示", "CRF 引擎未加载成功，请检查模型与词表文件。")
                return
        else:
            if not self.segmenter:
                QMessageBox.warning(self, "提示", "DAG+HMM 引擎未就绪，请确认默认词典文件存在。")
                return
            
        raw_text = self.text_input.toPlainText().strip()
        if not raw_text:
            QMessageBox.information(self, "提示", "请先输入或导入需要分词的文本！")
            return
            
        # 核心动态调整：同步 GUI 复选框状态到引擎
        mode_text = "CRF 神经网络引擎"
        if self.cmb_engine.currentText() == "DAG+HMM":
            self.segmenter.hmm_enabled = self.chk_hmm.isChecked()
            mode_text = "混合引擎 (DAG+HMM)" if self.segmenter.hmm_enabled else "纯净引擎 (纯 DAG)"
        self.status_bar.showMessage(f"🔪 正在使用 {mode_text} 进行运算...")
        self.btn_segment.setEnabled(False)
        QApplication.processEvents()
        
        try:
            lines = raw_text.split('\n')
            result_lines = []
            
            for line in lines:
                if not line.strip():
                    result_lines.append("")
                    continue
                if self.cmb_engine.currentText() == "CRF":
                    words = self.crf_segmenter.cut(line)
                else:
                    words = self.segmenter.cut(line)
                result_lines.append(" / ".join(words))
                
            final_result = "\n".join(result_lines)
            self.text_output.setPlainText(final_result)
            self.status_bar.showMessage(f"✨ 分词完成！({mode_text})")
            
        except Exception as e:
            QMessageBox.critical(self, "分词错误", f"运算过程中发生异常:\n{str(e)}")
            self.status_bar.showMessage("❌ 分词失败")
        finally:
            self.btn_segment.setEnabled(True)

    def export_file(self):
        result_text = self.text_output.toPlainText()
        if not result_text:
            QMessageBox.warning(self, "提示", "目前没有可导出的分词结果！")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "保存分词结果", self.base_dir, "Text Files (*.txt)")
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(result_text)
                self.status_bar.showMessage(f"💾 分词结果已保存至: {os.path.basename(file_path)}")
                QMessageBox.information(self, "成功", "分词结果导出成功！")
            except Exception as e:
                QMessageBox.warning(self, "导出失败", f"无法保存文件:\n{str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    
    window = CkCutWindow()
    window.show()
    
    sys.exit(app.exec())