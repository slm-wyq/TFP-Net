import pandas as pd
from glyles import convert
import argparse


def convert_glycans_to_smiles(input_file, output_file, glycan_column='target', smiles_column='smiles'):
    """
    将 CSV 文件中的聚糖转换为 SMILES 格式

    参数:
    input_file (str): 输入 CSV 文件路径
    output_file (str): 输出 CSV 文件路径
    glycan_column (str): 包含聚糖数据的列名
    smiles_column (str): 要添加的 SMILES 列名
    """
    try:
        # 读取 CSV 文件
        df = pd.read_csv(input_file)

        # 检查聚糖列是否存在
        if glycan_column not in df.columns:
            raise ValueError(f"列 '{glycan_column}' 不存在于 CSV 文件中")

        # 创建新列并填充 SMILES 值
        print(f"正在转换 {len(df)} 条聚糖数据...")

        # 修正：提取 SMILES 字符串而非整个元组
        df[smiles_column] = df[glycan_column].apply(lambda x: convert(glycan=x)[0][1] if convert(glycan=x) else '')

        # 保存结果
        df.to_csv(output_file, index=False)
        print(f"转换完成，结果已保存至 {output_file}")

    except Exception as e:
        print(f"发生错误: {e}")
        return False

    return True

    # 执行转换
success = convert_glycans_to_smiles(
    '../data/MPM-fingerprint.csv',
    '../output/MPM-smiles.csv',
    glycan_column='target',
    smiles_column='smiles'
)

if success:
    print("程序成功执行完毕")
else:
    print("程序执行失败")

