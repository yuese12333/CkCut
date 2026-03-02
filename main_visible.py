import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QTextEdit, QPushButton, QLabel,
                            QFileDialog, QMessageBox, QStatusBar, QFrame,
                            QCheckBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

# 导入我们的分词引擎
from src.segmenter import AutoSegmenter

class CkCutWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 默认词典与 HMM 模型路径
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dict_path = os.path.join(self.base_dir, "data", "output_dict", "my_dict_wiki.txt")
        self.hmm_path = os.path.join(self.base_dir, "data", "output_dict", "hmm_model.json")
        self.segmenter = None
        
        self.init_ui()
        self.load_engine()

    def init_ui(self):
        """初始化 UI 布局和控件"""
        self.setWindowTitle("🚀 CkCut 智能分词体验终端 v2.2")
        self.resize(950, 650)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # ================= 上方：双排控制台 =================
        
        # --- 第一排：词典引擎管理 ---
        dict_layout = QHBoxLayout()
        self.btn_import_dict = QPushButton("📥 挂载新词典")
        self.btn_export_dict = QPushButton("📤 导出当前词典")
        self.lbl_current_dict = QLabel("当前词典: 未加载")
        self.lbl_current_dict.setStyleSheet("color: #555; font-style: italic;")
        
        dict_layout.addWidget(self.btn_import_dict)
        dict_layout.addWidget(self.btn_export_dict)
        dict_layout.addWidget(self.lbl_current_dict)
        dict_layout.addStretch() 
        
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        
        # --- 第二排：文本与分词操作 ---
        text_op_layout = QHBoxLayout()
        self.btn_import = QPushButton("📂 智能导入 TXT")
        
        # HMM 兜底开关
        self.chk_hmm = QCheckBox("✨ 启用 HMM 兜底 (识别未登录词)")
        self.chk_hmm.setChecked(True)
        self.chk_hmm.setToolTip("开启后将使用 Viterbi 算法智能缝合未被收录的陌生词汇。")
        self.chk_hmm.setStyleSheet("font-weight: bold; color: #d35400;")
        
        self.btn_segment = QPushButton("🔪 界面一键分词")
        self.btn_export = QPushButton("💾 导出界面结果")
        
        self.btn_segment.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        
        text_op_layout.addWidget(self.btn_import)
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
        self.btn_import_dict.clicked.connect(self.import_dict)
        self.btn_export_dict.clicked.connect(self.export_dict)
        self.btn_import.clicked.connect(self.import_file)
        self.btn_segment.clicked.connect(self.run_segmentation)
        self.btn_export.clicked.connect(self.export_file)

    # ---------------- 词典引擎逻辑 ----------------

    def load_engine(self):
        """加载分词引擎主词典与 HMM 模型"""
        if not os.path.exists(self.dict_path):
            self.lbl_current_dict.setText(f"当前词典: 找不到文件 ({os.path.basename(self.dict_path)})")
            self.lbl_current_dict.setStyleSheet("color: red; font-weight: bold;")
            self.status_bar.showMessage("❌ 引擎加载失败：缺失主词典")
            self.btn_segment.setEnabled(False)
            return
            
        self.status_bar.showMessage(f"⏳ 正在加载混合分词引擎，请稍候...")
        self.lbl_current_dict.setText(f"当前词典: {os.path.basename(self.dict_path)}")
        self.lbl_current_dict.setStyleSheet("color: green; font-weight: bold;")
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
                
            self.status_bar.showMessage(f"✅ 双引擎就绪！DAG 节点数: {len(self.segmenter.FREQ)}")
            self.btn_segment.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"词典加载异常: {str(e)}")
            self.status_bar.showMessage("❌ 引擎加载异常")

    def import_dict(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择分词词典", self.base_dir, "Text Files (*.txt);;All Files (*)")
        if file_path:
            self.dict_path = file_path
            self.load_engine()
            QMessageBox.information(self, "成功", f"成功挂载新词典:\n{os.path.basename(file_path)}")

    def export_dict(self):
        if not self.segmenter or not self.segmenter.FREQ:
            QMessageBox.warning(self, "提示", "当前没有加载任何有效的词典！")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "导出当前词典", self.base_dir, "Text Files (*.txt)")
        if file_path:
            try:
                self.status_bar.showMessage("⏳ 正在整理并导出词典...")
                QApplication.processEvents()
                
                valid_words = [(word, freq) for word, freq in self.segmenter.FREQ.items() if freq > 0]
                valid_words.sort(key=lambda x: x[1], reverse=True)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    for word, freq in valid_words:
                        f.write(f"{word} {freq}\n")
                        
                self.status_bar.showMessage(f"💾 词典已成功导出至: {os.path.basename(file_path)}")
                QMessageBox.information(self, "成功", f"成功导出 {len(valid_words)} 个词条！")
            except Exception as e:
                QMessageBox.warning(self, "导出失败", f"无法保存词典:\n{str(e)}")

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
        if not self.segmenter:
            QMessageBox.warning(self, "提示", "请先挂载有效的词典！")
            return
            
        save_path, _ = QFileDialog.getSaveFileName(self, "保存分词结果", self.base_dir, "Text Files (*.txt)")
        if not save_path:
            return
            
        self.status_bar.showMessage("🔪 正在后台流式分词中，请耐心等候...")
        # 冻结所有可能干扰后台进程的按钮
        self.btn_segment.setEnabled(False)
        self.btn_import.setEnabled(False)
        self.btn_import_dict.setEnabled(False)
        self.chk_hmm.setEnabled(False)
        QApplication.processEvents()
        
        # 同步当前的 HMM 开关状态
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
            self.btn_import_dict.setEnabled(True)
            self.chk_hmm.setEnabled(True)

    # ---------------- 界面文本分词逻辑 ----------------

    def run_segmentation(self):
        if not self.segmenter:
            QMessageBox.warning(self, "提示", "请先挂载有效的词典！")
            return
            
        raw_text = self.text_input.toPlainText().strip()
        if not raw_text:
            QMessageBox.information(self, "提示", "请先输入或导入需要分词的文本！")
            return
            
        # 核心动态调整：同步 GUI 复选框状态到引擎
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