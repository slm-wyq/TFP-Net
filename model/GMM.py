import re
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

# ===================== 配置 =====================
TRAIN_CSV   = "../data/glycan_embeddings.csv"             # 参考（训练）嵌入
TRAIN_OUT   = "../output/glycan_embeddings_with_risk_best.csv"
ARTIFACT    = "risk_model.joblib"                 # 保存训练好的全部对象

SCORING_METHOD = "gmm"        # 'gmm' 或 'kde'
WITH_ZSCORE    = True         # 降维前是否做 z-score（高维嵌入建议 True）
PCA_DIM        = 20           # 先 PCA 白化到多少维；None=不降维。KDE 建议更小(≈10)
RANDOM_STATE   = 42

# GMM 相关
GMM_N_COMPONENTS = None       # None=按 BIC 自动选；或直接给 3 / 5 锁定
GMM_CANDIDATES   = [1, 2, 3, 4, 5]
COV_TYPE         = "diag"     # 'diag' 稳健；样本充足且 PCA_DIM 较小(≤10)时可用 'full'

# KDE 相关
KDE_BANDWIDTH = "cv"          # 'cv'(交叉验证) / 'scott' / 或一个浮点数
# =================================================


# --------------------- 工具函数 ---------------------
def get_embedding_cols(df):
    pat = re.compile(r"^embedding_(\d+)$")
    cols = [(int(m.group(1)), c) for c in df.columns if (m := pat.match(c))]
    if not cols:
        raise ValueError("未发现 embedding_* 列。")
    return [c for _, c in sorted(cols, key=lambda x: x[0])]


def fit_gmm_by_bic(Xg, candidates, cov_type, random_state, group=""):
    """在候选 n_components 中按 BIC 选最优单/多组分 GMM。"""
    n, d = Xg.shape
    # 每个组分所需的最少样本数（随协方差复杂度变化），避免过拟合 / 奇异协方差
    per_comp = {"full": 8 * d, "tied": 4 * d, "diag": 8, "spherical": 5}.get(cov_type, 10)
    best = None
    for k in candidates:
        if k > 1 and n < per_comp * k:      # 守卫：样本不足以支撑这么多组分
            continue
        gmm = GaussianMixture(
            n_components=k, covariance_type=cov_type, random_state=random_state,
            reg_covar=1e-5, n_init=3, max_iter=500,
        ).fit(Xg)
        bic = gmm.bic(Xg)
        if (best is None) or (bic < best[1]):
            best = (gmm, bic, k)
    print(f"[GMM:{group}] 选定 n_components={best[2]}  (BIC={best[1]:.1f}, 样本={n}, cov={cov_type})")
    return best[0]


def fit_density(Xg, group=""):
    """根据 SCORING_METHOD 返回一个带 .score_samples() 的密度模型。"""
    if Xg.shape[0] < 5:
        raise ValueError(f"[{group}] 样本太少（{Xg.shape[0]}），无法拟合密度。")

    if SCORING_METHOD == "gmm":
        if GMM_N_COMPONENTS is not None:
            gmm = GaussianMixture(
                n_components=GMM_N_COMPONENTS, covariance_type=COV_TYPE,
                random_state=RANDOM_STATE, reg_covar=1e-5, n_init=3, max_iter=500,
            ).fit(Xg)
            print(f"[GMM:{group}] 手动锁定 n_components={GMM_N_COMPONENTS} "
                  f"(BIC={gmm.bic(Xg):.1f})")
            return gmm
        return fit_gmm_by_bic(Xg, GMM_CANDIDATES, COV_TYPE, RANDOM_STATE, group)

    elif SCORING_METHOD == "kde":
        if KDE_BANDWIDTH == "cv":
            from sklearn.model_selection import GridSearchCV
            grid = GridSearchCV(
                KernelDensity(kernel="gaussian"),
                {"bandwidth": np.logspace(-1.0, 0.6, 16)},
                cv=min(5, Xg.shape[0]),
            ).fit(Xg)
            bw = grid.best_params_["bandwidth"]
        elif KDE_BANDWIDTH in ("scott", "silverman"):
            n, d = Xg.shape
            bw = n ** (-1.0 / (d + 4))     # 白化后各维方差≈1，可用各向同性带宽
        else:
            bw = float(KDE_BANDWIDTH)
        kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(Xg)
        print(f"[KDE:{group}] bandwidth={bw:.4f}  (样本={Xg.shape[0]})")
        return kde

    else:
        raise ValueError("SCORING_METHOD 只能是 'gmm' 或 'kde'。")


def build_preprocessor(X):
    """在【全部参考嵌入】上拟合 scaler 与 PCA，保证两类共享同一投影。"""
    scaler = StandardScaler().fit(X) if WITH_ZSCORE else None
    Xs = scaler.transform(X) if scaler is not None else X
    if PCA_DIM is not None:
        pca = PCA(n_components=PCA_DIM, whiten=True, random_state=RANDOM_STATE).fit(Xs)
        ev = pca.explained_variance_ratio_.sum()
        print(f"[PCA] {X.shape[1]} → {PCA_DIM} 维（白化），累计解释方差 {ev:.1%}")
    else:
        pca = None
    return scaler, pca


def transform(X, scaler, pca):
    Xs = scaler.transform(X) if scaler is not None else X
    return pca.transform(Xs) if pca is not None else Xs


# --------------------- 训练（仅训练，移除全部预测逻辑） ---------------------
def fit_reference():
    df = pd.read_csv(TRAIN_CSV)
    emb_cols = get_embedding_cols(df)
    X = df[emb_cols].values
    print(f"[INFO] 参考样本: {X.shape[0]}，嵌入维度: {X.shape[1]}")

    for col in ("Host_Pathogen", "immunogenicity"):
        if col not in df.columns:
            raise ValueError(f"缺少 {col} 列。")

    human_mask = (df["Host_Pathogen"] == "Homo_sapiens").values
    patho_imm_mask = ((df["Host_Pathogen"] == "Pathogen") &
                      (df["immunogenicity"] == 1)).values
    print(f"[INFO] 人类: {human_mask.sum()}，免疫原性病原体: {patho_imm_mask.sum()}")

    # 预处理（在全体参考嵌入上拟合）
    scaler, pca = build_preprocessor(X)
    Xp = transform(X, scaler, pca)

    # 两类参考分布
    model_host = fit_density(Xp[human_mask], group="host")
    model_pi   = fit_density(Xp[patho_imm_mask], group="pathogen_imm")

    # 打分（对全部参考样本）
    logp_h = model_host.score_samples(Xp)
    logp_p = model_pi.score_samples(Xp)
    score  = logp_h - logp_p

    df["logp_human"] = logp_h
    df["logp_pathogen_immune"] = logp_p
    df["Risk_LLR_immunePathogen"] = score
    df["Risk_percentile"] = pd.Series(score).rank(pct=True).values   # 越大越像宿主
    # 有界映射到 (-1,1)，k 由参考分布尺度自适应（替代旧的固定 0.0004）
    k = 1.0 / (np.std(score) + 1e-12)
    df["Risk_score_bounded"] = np.tanh(k * score)

    df.to_csv(TRAIN_OUT, index=False)
    print(f"[DONE] 参考结果写出：{TRAIN_OUT}")

    # 保存全部训练对象（不含预测推理代码，仅权重/预处理/参考分布）
    dump({
        "scaler": scaler, "pca": pca,
        "model_host": model_host, "model_pi": model_pi,
        "ref_score_sorted": np.sort(score), "k": k,
        "emb_cols": emb_cols, "method": SCORING_METHOD,
    }, ARTIFACT)
    print(f"[DONE] 模型与预处理已保存：{ARTIFACT}")


if __name__ == "__main__":
    # 仅执行训练，删除新样本预测调用
    fit_reference()