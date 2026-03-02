import os
import regex as re
from tqdm import tqdm
from pathlib import Path

def clean_and_split_text(text: str) -> list:
    """
    清洗并切分文本。
    利用正则表达式匹配所有连续的汉字块，非汉字字符（如标点、英文、数字）将被视为天然的分隔符。
    
    :param text: 原始文本行
    :return: 仅包含纯汉字短句的列表
    """
    pattern = re.compile(r'\p{Han}+')
    fragments = pattern.findall(text)
    return fragments

def preprocess_directory(input_dir: str, output_filepath: str, min_length: int = 2):
    """
    遍历指定目录下的所有 .txt 文件，进行清洗并将结果合并写入到一个输出文件中。
    
    :param input_dir: 存放原始语料的文件夹路径
    :param output_filepath: 清洗合并后的单一输出文件路径
    :param min_length: 保留的最短句子长度
    """
    input_path = Path(input_dir)
    
    # 检查输入目录是否存在
    if not input_path.exists() or not input_path.is_dir():
        print(f"❌ 错误: 找不到输入目录 {input_dir}")
        return

    # 找出目录下所有的 .txt 文件 (rglob 支持递归查找子文件夹)
    txt_files = list(input_path.rglob("*.txt"))
    if not txt_files:
        print(f"⚠️ 警告: 在 {input_dir} 下没有找到任何 .txt 文件。")
        return

    # 确保输出文件所在的目录存在
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    print(f"🚀 开始预处理目录 '{input_dir}' 下的 {len(txt_files)} 个文件...")

    total_lines = 0
    valid_fragments = 0

    # 打开统一的输出文件
    with open(output_filepath, 'w', encoding='utf-8') as fout:
        
        # 【修改核心】：在外层使用单一总进度条，以“文件”为单位
        # 将 txt_files 传入 tqdm，自动推断总量
        with tqdm(txt_files, desc="批量预处理总进度", unit="文件") as pbar:
            for file_path in pbar:
                # 在进度条尾部动态显示当前正在处理的文件名，避免 print 刷屏
                pbar.set_postfix(file=file_path.name)
                
                # 读取当前文件
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as fin:
                    for line in fin:
                        total_lines += 1
                        
                        line = line.strip()
                        if not line:
                            continue
                        
                        # 提取纯中文短句
                        fragments = clean_and_split_text(line)
                        
                        # 写入合并后的输出文件
                        for frag in fragments:
                            if len(frag) >= min_length:
                                fout.write(frag + '\n')
                                valid_fragments += 1

    print(f"\n✅ 所有文件预处理完成！")
    print(f"📊 统计信息: 共读取 {total_lines} 行原始文本，生成了 {valid_fragments} 句纯中文有效短句。")
    print(f"📁 合并后的清洗数据已保存至: {output_filepath}")

if __name__ == "__main__":
    # 动态获取项目根目录，确保无论在哪个路径下运行脚本，都能找对 data 文件夹
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 构建绝对路径
    input_directory = os.path.join(BASE_DIR, "data", "raw_corpus")
    output_file = os.path.join(BASE_DIR, "data", "processed", "merged_cleaned.txt")
    
    # 执行批量预处理
    preprocess_directory(input_directory, output_file)