import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILE_PATH = "../output/glycan_embeddings_with_risk_best.csv"  
k = 1e-1                      
SAVE_PATH = "../output/sigmod_risk.csv"   


df = pd.read_csv(FILE_PATH)


x = df["Risk_LLR_immunePathogen"].values


def scale_to_neg1_1(x, k=1e-4):
    return 2 / (1 + np.exp(-k * x)) - 1


df["Risk_Scaled_-1_1"] = scale_to_neg1_1(x, k=k)


df.to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")


plt.figure(figsize=(10, 4))

plt.subplot(1,2,1)
plt.hist(x, bins=30, color='salmon', alpha=0.7)
plt.title(f"原始分数\n范围: [{x.min():.2f}, {x.max():.2f}]")

plt.subplot(1,2,2)
scaled = df["Risk_Scaled_-1_1"]
plt.hist(scaled, bins=30, color='lightblue', alpha=0.7)
plt.title(f"标准化到 [-1,1]\n范围: [{scaled.min():.3f}, {scaled.max():.3f}]")
plt.axvline(0, color='red', linestyle='--', label='0点不变')
plt.legend()

plt.tight_layout()
plt.show()

print("✅ 处理完成！")
print(f"原始分数范围：{x.min():.2f} ~ {x.max():.2f}")
print(f"标准化后范围：{scaled.min():.3f} ~ {scaled.max():.3f}")
print(f"结果已保存到：{SAVE_PATH}")
