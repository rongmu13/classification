import io
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt


st.set_page_config(page_title="🧪 表形式データ分類アプリ（Excel）", layout="wide")
st.title("🧪 表形式データ分類アプリ（Excel）")

st.markdown("""
**使い方**  
1) Excel をアップロード（**1列目＝元ラベル**、以降＝数値の特徴量）。  
2) 目的クラス数（2/3/4/5）を選び、**「元ラベル → 新ラベル(0..K-1)」** を割り当て。  
3) 使う特徴量・分類器・PCA の有無を選ぶ。  
4) **「学習して評価」** をクリック。  
""")

# ---- サイドバー設定 ----
st.sidebar.header("⚙️ 設定")

test_size = st.sidebar.slider("テストデータ割合", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("ランダムシード (random_state)", value=42, step=1)
use_pca = st.sidebar.checkbox("PCA を使用（木系モデル以外）", value=False)
pca_var = st.sidebar.slider("PCA 累積寄与率", 0.80, 0.99, 0.95, 0.01)

clf_name = st.sidebar.selectbox(
    "分類器を選択",
    ["KNN", "線形SVM", "SVM（RBF）", "ロジスティック回帰", "ランダムフォレスト"]
)

# ハイパーパラメータ
if clf_name == "KNN":
    knn_k = st.sidebar.slider("KNN: 近傍数 (n_neighbors)", 1, 25, 5, 1)
elif clf_name == "線形SVM":
    linsvm_C = st.sidebar.slider("線形SVM: C", 0.01, 10.0, 1.0, 0.01)
elif clf_name == "SVM（RBF）":
    rbf_C = st.sidebar.slider("SVM（RBF）: C", 0.01, 10.0, 1.0, 0.01)
elif clf_name == "ロジスティック回帰":
    logreg_C = st.sidebar.slider("ロジスティック回帰: C", 0.01, 10.0, 1.0, 0.01)
elif clf_name == "ランダムフォレスト":
    rf_n = st.sidebar.slider("ランダムフォレスト: n_estimators", 50, 500, 300, 50)
    rf_depth = st.sidebar.slider("ランダムフォレスト: max_depth（None=0）", 0, 50, 0, 1)

st.sidebar.divider()
st.sidebar.caption("欠損値は列の中央値で補完。非木系モデルは標準化を実施。クラス不均衡では多くのモデルで class_weight='balanced' を使用（KNN を除く）。")

# ---- ファイル入力 ----
uploaded = st.file_uploader("📤 Excel ファイル（.xlsx）をアップロード", type=["xlsx"])
if uploaded is None:
    st.info("Excel をアップロードしてください。**1列目は元ラベル**、以降は数値の特徴量です。")
    st.stop()

# Excel 読み込み
try:
    bytes_data = uploaded.read()
    xls = pd.ExcelFile(io.BytesIO(bytes_data))
    sheet_name = st.selectbox("ワークシートを選択", xls.sheet_names)
    df = pd.read_excel(io.BytesIO(bytes_data), sheet_name=sheet_name)
except Exception as e:
    st.error(f"Excel の読み込みに失敗しました：{e}")
    st.stop()

if df.shape[1] < 2:
    st.error("少なくとも2列が必要です：1列目＝元ラベル、その後は特徴量。")
    st.stop()

st.subheader("データプレビュー")
st.dataframe(df.head(), use_container_width=True)

# 列の識別
orig_label_col = df.columns[0]
feature_cols_all = [c for c in df.columns[1:]]

# 特徴量を数値へ強制変換（非数値は NaN へ）
df_features = df[feature_cols_all].apply(pd.to_numeric, errors="coerce")

# 欠損値レポート
with st.expander("🔎 欠損値レポート"):
    na_counts = df_features.isna().sum()
    st.write(na_counts.to_frame("NaN 件数"))
    st.caption("特徴量の非数値・空白は NaN に変換され、後で中央値で補完します。")

# ---- ラベルマッピング ----
st.subheader("🎯 ラベルのマッピングと分類設定")
unique_orig_labels = pd.Series(df[orig_label_col].unique()).tolist()
unique_orig_labels_sorted = sorted(unique_orig_labels, key=lambda x: str(x))

num_classes = st.radio("目的クラス数", [2, 3, 4, 5], horizontal=True)

st.write("各 **元ラベル値** に対して、新しい **目的ラベル（0..K-1）** を割り当ててください：")
mapping: Dict = {}
cols = st.columns(2)
for idx, val in enumerate(unique_orig_labels_sorted):
    with cols[idx % 2]:
        new_lab = st.selectbox(
            f"元ラベル {val} →",
            options=list(range(num_classes)),
            key=f"map_{idx}",
            index=0
        )
        mapping[val] = new_lab

st.caption(f"現在のマッピング：{mapping}")

# 新ラベルの作成
try:
    y_new = df[orig_label_col].map(mapping)
except Exception as e:
    st.error(f"ラベルのマッピングに失敗しました：{e}")
    st.stop()

mask_valid = y_new.notna()
dropped = (~mask_valid).sum()
if dropped > 0:
    st.warning(f"{dropped} 行でラベルがマッピングできなかったため削除しました。マッピングを確認してください。")
y = y_new[mask_valid].astype(int).values
X_all = df_features.loc[mask_valid].copy()

# ---- 特徴量選択 ----
st.subheader("🧩 特徴量の選択")
chosen_feats = st.multiselect(
    "学習に使用する特徴量列を選択（複数選択可）",
    options=feature_cols_all,
    default=feature_cols_all
)

if len(chosen_feats) == 0:
    st.error("少なくとも1つの特徴量を選択してください。")
    st.stop()

X = X_all[chosen_feats].values

# 学習/テスト分割（層化）
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
except ValueError as e:
    st.error(f"データ分割に失敗しました：{e}（あるクラスのサンプル数が少なすぎる可能性）")
    st.stop()

# ---- パイプライン ----
num_classes_int = int(num_classes)
average_mode = "binary" if num_classes_int == 2 else "macro"
average_mode_jp = "二値" if average_mode == "binary" else "マクロ"

def make_non_tree_pipeline(base_estimator):
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if use_pca and clf_name != "ランダムフォレスト":
        steps.append(("pca", PCA(n_components=pca_var, svd_solver="full")))
    steps.append(("clf", base_estimator))
    return Pipeline(steps)

def make_tree_pipeline(base_estimator):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", base_estimator)
    ])

# 分類器の生成
if clf_name == "KNN":
    clf = make_non_tree_pipeline(KNeighborsClassifier(n_neighbors=knn_k))
elif clf_name == "線形SVM":
    clf = make_non_tree_pipeline(LinearSVC(C=linsvm_C, class_weight="balanced", dual="auto"))
elif clf_name == "SVM（RBF）":
    clf = make_non_tree_pipeline(SVC(C=rbf_C, kernel="rbf", class_weight="balanced", probability=True))
elif clf_name == "ロジスティック回帰":
    clf = make_non_tree_pipeline(LogisticRegression(C=logreg_C, max_iter=2000, class_weight="balanced"))
elif clf_name == "ランダムフォレスト":
    md = None if rf_depth == 0 else rf_depth
    clf = make_tree_pipeline(RandomForestClassifier(
        n_estimators=rf_n,
        max_depth=md,
        random_state=random_state,
        class_weight="balanced"
    ))
else:
    st.error("未知の分類器です。")
    st.stop()

# ---- 学習と評価 ----
st.subheader("🚀 学習と評価")
if st.button("学習して評価", type="primary"):
    try:
        clf.fit(X_train, y_train)
    except Exception as e:
        st.error(f"学習に失敗しました：{e}")
        st.stop()

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=average_mode)

    left, right, right2 = st.columns(3)
    with left:
        st.metric("正解率 (Accuracy)", f"{acc:.4f}")
    with right:
        st.metric(f"F1（{average_mode_jp}）", f"{f1:.4f}")

    # 二値分類のみ ROC AUC
    if num_classes_int == 2:
        try:
            if hasattr(clf, "decision_function"):
                y_score = clf.decision_function(X_test)
            elif hasattr(clf, "predict_proba"):
                y_score = clf.predict_proba(X_test)[:, 1]
            else:
                y_score = None
            if y_score is not None:
                auc = roc_auc_score(y_test, y_score)
                with right2:
                    st.metric("ROC AUC", f"{auc:.4f}")
        except Exception:
            pass
    else:
        st.caption("多クラスでは AUC は未計算です（OVO/OVR に拡張可能）。")

    # 混同行列
    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes_int)))
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(num_classes_int),
        yticks=np.arange(num_classes_int),
        xticklabels=[f"Pred {i}" for i in range(num_classes_int)],
        yticklabels=[f"True {i}" for i in range(num_classes_int)],
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix"
    )
    for i in range(num_classes_int):
        for j in range(num_classes_int):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    st.pyplot(fig)

    # 分類レポート（英語出力：sklearn 標準）
    st.subheader("📄 分類レポート")
    cr = classification_report(y_test, y_pred, digits=4, zero_division=0)
    st.code(cr, language="text")

    # 解釈可能性
    st.subheader("🧭 解釈可能性")
    try:
        if clf_name == "ランダムフォレスト":
            final = clf.named_steps["clf"]
            importances = final.feature_importances_
            imp_df = pd.DataFrame({"feature": chosen_feats, "importance": importances}).sort_values("importance", ascending=False)
            st.dataframe(imp_df, use_container_width=True)
            fig2, ax2 = plt.subplots(figsize=(6, min(0.35*len(chosen_feats)+1, 10)))
            ax2.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
            ax2.set_title("Feature Importance (Random Forest)")
            st.pyplot(fig2)

        elif clf_name in ["線形SVM", "ロジスティック回帰"]:
            final = clf.named_steps["clf"]
            if hasattr(final, "coef_"):
                coef = final.coef_
                if num_classes_int == 2:
                    coef = coef[0]
                    names = chosen_feats
                    if use_pca:
                        st.warning("PCA を使用中：係数は低次元空間のものです。元の特徴量への直接的な対応は弱くなります。")
                    coef_df = pd.DataFrame({"feature": names, "coef": coef if len(coef)==len(names) else np.nan})
                    coef_df = coef_df.sort_values("coef", key=np.abs, ascending=False)
                    st.dataframe(coef_df, use_container_width=True)
                    fig3, ax3 = plt.subplots(figsize=(6, min(0.35*len(chosen_feats)+1, 10)))
                    ax3.barh(coef_df["feature"][::-1], coef_df["coef"][::-1])
                    ax3.set_title("Linear Model Coefficients")
                    st.pyplot(fig3)
                else:
                    st.caption("多クラスの係数は one-vs-rest の行列です（表で確認できます）。")
                    st.dataframe(pd.DataFrame(coef, columns=chosen_feats), use_container_width=True)
            else:
                st.caption("このモデルには係数情報がありません。")
        else:
            st.caption("KNN / SVM（RBF）には直接の特徴量重要度はありません。")

    except Exception as e:
        st.warning(f"解釈可能性の計算で問題が発生しました：{e}")

st.divider()
st.caption("ヒント：あるクラスのテストサンプルが 0 になる場合、テスト割合を小さくするか、クラスを統合してください。")




