import os
import regex as re
from tqdm import tqdm

def clean_and_split_text(text: str) -> list:
    """
    清洗并切分文本。
    利用正则表达式匹配所有连续的汉字块，非汉字字符（如标点、英文、数字）将被视为天然的分隔符。
    
    :param text: 原始文本行
    :return: 仅包含纯汉字短句的列表
    """
    # \p{Han} 是 regex 库支持的 Unicode 属性，完美匹配所有中文字符
    # + 表示匹配一个或多个连续的中文字符
    pattern = re.compile(r'\p{Han}+')
    
    # findall 会返回所有匹配的连续汉字片段，自动丢弃了非汉字字符
    fragments = pattern.findall(text)
    
    return fragments

def preprocess_corpus(input_filepath: str, output_filepath: str, min_length: int = 2):
    """
    流式处理大规模语料库。
    逐行读取以节省内存，清洗后将合格的短句写入输出文件，每行一句。
    
    :param input_filepath: 原始语料文件路径 (TXT格式)
    :param output_filepath: 清洗后的输出文件路径
    :param min_length: 保留的最短句子长度（默认为2，因为单字无法形成词汇组合）
    """
    if not os.path.exists(input_filepath):
        print(f"❌ 错误: 找不到输入文件 {input_filepath}")
        return

    # 获取文件总字节数，用于 tqdm 进度条显示
    total_size = os.path.getsize(input_filepath)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    print(f"🚀 开始预处理语料: {input_filepath}")
    
    # 计数器
    total_lines = 0
    valid_fragments = 0

    # 打开文件，使用 utf-8 编码。遇到无法解析的字符直接忽略 (errors='ignore')
    with open(input_filepath, 'r', encoding='utf-8', errors='ignore') as fin, \
        open(output_filepath, 'w', encoding='utf-8') as fout:
        
        # 使用 tqdm 包装，以字节为单位显示进度
        with tqdm(total=total_size, desc="处理进度", unit='B', unit_scale=True) as pbar:
            for line in fin:
                # 更新进度条 (按行所占的字节数)
                pbar.update(len(line.encode('utf-8')))
                total_lines += 1
                
                # 去除行首尾空白字符
                line = line.strip()
                if not line:
                    continue
                
                # 提取纯中文短句
                fragments = clean_and_split_text(line)
                
                # 遍历提取出的片段，过滤掉太短的，然后写入新文件
                for frag in fragments:
                    if len(frag) >= min_length:
                        fout.write(frag + '\n')
                        valid_fragments += 1

    print(f"\n✅ 预处理完成！")
    print(f"📊 统计信息: 读取了 {total_lines} 行原始文本，生成了 {valid_fragments} 句纯中文有效短句。")
    print(f"📁 输出文件已保存至: {output_filepath}")

# 简单的测试入口
if __name__ == "__main__":
    # 假设你在项目根目录下运行此脚本
    # 我们先在 data/raw_corpus/ 下建一个 dummy_test.txt 做个小测试
    
    # 1. 模拟生成一个测试文件
    test_raw_dir = "../data/raw_corpus"
    test_out_dir = "../data/processed"
    os.makedirs(test_raw_dir, exist_ok=True)
    
    test_file = os.path.join(test_raw_dir, "dummy_test.txt")
    out_file = os.path.join(test_out_dir, "cleaned_corpus.txt")
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("大家好！这是一个从零开始的 NLP 项目。AutoSeg v1.0 棒极了，不是吗？\n")
        f.write("We need to handle English characters 还有各种数字123和标点符号...\n")
        f.write("一！\n") # 这句太短会被过滤掉
        
    # 2. 运行预处理
    preprocess_corpus(test_file, out_file)
    
    # 3. 打印结果看看效果
    print("\n🧐 预览处理结果:")
    with open(out_file, 'r', encoding='utf-8') as f:
        print(f.read())