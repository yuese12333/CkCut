import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QTextEdit, QPushButton, QLabel,
                            QFileDialog, QMessageBox, QStatusBar, QFrame)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

# 导入我们的分词引擎
from src.segmenter import AutoSegmenter

class AutoSegWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 默认词典路径
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dict_path = os.path.join(self.base_dir, "data", "output_dict", "my_dict_wiki.txt")
        self.segmenter = None
        
        self.init_ui()
        self.load_engine()

    def init_ui(self):
        """初始化 UI 布局和控件"""
        self.setWindowTitle("🚀 AutoSeg 智能分词体验终端 v2.0")
        self.resize(900, 650)
        
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
        dict_layout.addStretch() # 把文字推到左边
        
        # 添加一条分割线
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        
        # --- 第二排：文本与分词操作 ---
        text_op_layout = QHBoxLayout()
        self.btn_import = QPushButton("📂 导入待分词 TXT")
        self.btn_segment = QPushButton("🔪 一键智能分词")
        self.btn_export = QPushButton("💾 导出分词结果")
        
        self.btn_segment.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        
        text_op_layout.addWidget(self.btn_import)
        text_op_layout.addWidget(self.btn_segment)
        text_op_layout.addWidget(self.btn_export)
        
        # 将两排按钮加入主布局
        main_layout.addLayout(dict_layout)
        main_layout.addWidget(line)
        main_layout.addLayout(text_op_layout)
        
        # ================= 下方：双栏文本区 =================
        text_layout = QHBoxLayout()
        
        # 左侧：原文输入框
        left_layout = QVBoxLayout()
        left_label = QLabel("📝 原始文本:")
        left_label.setFont(QFont("Microsoft YaHei", 10, QFont.Weight.Bold))
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("请输入需要分词的中文句子，或者点击上方按钮导入 txt 文件...")
        self.text_input.setFont(QFont("Microsoft YaHei", 11))
        left_layout.addWidget(left_label)
        left_layout.addWidget(self.text_input)
        
        # 右侧：分词结果框
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
        """加载分词引擎词典"""
        if not os.path.exists(self.dict_path):
            self.lbl_current_dict.setText(f"当前词典: 找不到文件 ({os.path.basename(self.dict_path)})")
            self.lbl_current_dict.setStyleSheet("color: red; font-weight: bold;")
            self.status_bar.showMessage("❌ 引擎加载失败：缺失词典")
            self.btn_segment.setEnabled(False)
            return
            
        self.status_bar.showMessage(f"⏳ 正在加载引擎，请稍候...")
        self.lbl_current_dict.setText(f"当前词典: {os.path.basename(self.dict_path)}")
        self.lbl_current_dict.setStyleSheet("color: green; font-weight: bold;")
        QApplication.processEvents() 
        
        try:
            self.segmenter = AutoSegmenter(self.dict_path)
            self.status_bar.showMessage(f"✅ 引擎就绪！有效节点数: {len(self.segmenter.FREQ)}")
            self.btn_segment.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"词典加载异常: {str(e)}")
            self.status_bar.showMessage("❌ 引擎加载异常")

    def import_dict(self):
        """允许用户动态导入本地的自定义词典"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择分词词典", self.base_dir, "Text Files (*.txt);;All Files (*)")
        if file_path:
            self.dict_path = file_path
            self.load_engine()
            QMessageBox.information(self, "成功", f"成功挂载新词典:\n{os.path.basename(file_path)}")

    def export_dict(self):
        """将当前引擎内存中的有效词汇(剔除前缀残留)导出为标准词典"""
        if not self.segmenter or not self.segmenter.FREQ:
            QMessageBox.warning(self, "提示", "当前没有加载任何有效的词典！")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "导出当前词典", self.base_dir, "Text Files (*.txt)")
        if file_path:
            try:
                self.status_bar.showMessage("⏳ 正在整理并导出词典...")
                QApplication.processEvents()
                
                # 过滤出真正的词 (我们的前缀树机制会把非词前缀的 freq 设为 0)
                valid_words = [(word, freq) for word, freq in self.segmenter.FREQ.items() if freq > 0]
                # 按词频降序排列，保证词典的整洁性
                valid_words.sort(key=lambda x: x[1], reverse=True)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    for word, freq in valid_words:
                        f.write(f"{word} {freq}\n")
                        
                self.status_bar.showMessage(f"💾 词典已成功导出至: {os.path.basename(file_path)}")
                QMessageBox.information(self, "成功", f"成功导出 {len(valid_words)} 个词条！")
            except Exception as e:
                QMessageBox.warning(self, "导出失败", f"无法保存词典:\n{str(e)}")

    # ---------------- 文本分词逻辑 ----------------

    def import_file(self):
        """导入 TXT 文件到左侧输入框"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文本文件", self.base_dir, "Text Files (*.txt);;All Files (*)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if len(content) > 100000:
                        reply = QMessageBox.warning(self, "文件过大", "该文件内容极大，直接显示可能会卡顿。是否仅截取前 10 万字显示？", 
                                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                        if reply == QMessageBox.StandardButton.Yes:
                            content = content[:100000]
                            
                    self.text_input.setPlainText(content)
                    self.status_bar.showMessage(f"📁 成功导入文本: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.warning(self, "导入失败", f"无法读取文件:\n{str(e)}")

    def run_segmentation(self):
        """执行分词逻辑"""
        if not self.segmenter:
            QMessageBox.warning(self, "提示", "请先挂载有效的词典！")
            return
            
        raw_text = self.text_input.toPlainText().strip()
        if not raw_text:
            QMessageBox.information(self, "提示", "请先输入或导入需要分词的文本！")
            return
            
        self.status_bar.showMessage("🔪 正在进行高能分词运算...")
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
            self.status_bar.showMessage("✨ 分词完成！")
            
        except Exception as e:
            QMessageBox.critical(self, "分词错误", f"运算过程中发生异常:\n{str(e)}")
            self.status_bar.showMessage("❌ 分词失败")
        finally:
            self.btn_segment.setEnabled(True)

    def export_file(self):
        """将右侧分词结果导出为 TXT"""
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
    
    window = AutoSegWindow()
    window.show()
    
    sys.exit(app.exec())