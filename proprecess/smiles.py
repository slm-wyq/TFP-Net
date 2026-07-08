import pandas as pd
from glyles import convert
import argparse


def convert_glycans_to_smiles(input_file, output_file, glycan_column='target', smiles_column='smiles'):
    try:
        df = pd.read_csv(input_file)

        if glycan_column not in df.columns:
            raise ValueError(f"列 '{glycan_column}' 不存在于 CSV 文件中")

        print(f"正在转换 {len(df)} 条聚糖数据...")

        df[smiles_column] = df[glycan_column].apply(lambda x: convert(glycan=x)[0][1] if convert(glycan=x) else '')

        df.to_csv(output_file, index=False)
        print(f"转换完成，结果已保存至 {output_file}")

    except Exception as e:
        print(f"发生错误: {e}")
        return False

    return True
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

