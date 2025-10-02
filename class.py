import io
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List

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


st.set_page_config(page_title="Tabular Classifier App", layout="wide")
st.title("🧪 Tabular Classifier（Excel 一键分类）")

st.markdown("""
**用法**  
1) 上传 Excel（第一列是**原始标签**，后面的列都是特征）。  
2) 选择目标分类数（2/3/4/5），把“原始标签 → 新标签(0..K-1)”一一映射。  
3) 选择要用的特征、分类器，是否用 PCA。  
4) 点“训练并评估”。  
""")

# --- Sidebar controls ---
st.sidebar.header("⚙️ 参数设置")

test_size = st.sidebar.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("random_state", value=42, step=1)
use_pca = st.sidebar.checkbox("使用 PCA（仅对非树模型）", value=False)
pca_var = st.sidebar.slider("PCA 保留方差比例", 0.80, 0.99, 0.95, 0.01)

clf_name = st.sidebar.selectbox(
    "选择分类器",
    ["KNN", "Linear SVM", "SVM (RBF)", "Logistic Regression", "Random Forest"]
)

# hyperparams
if clf_name == "KNN":
    knn_k = st.sidebar.slider("KNN: n_neighbors", 1, 25, 5, 1)
elif clf_name == "Linear SVM":
    linsvm_C = st.sidebar.slider("Linear SVM: C", 0.01, 10.0, 1.0, 0.01)
elif clf_name == "SVM (RBF)":
    rbf_C = st.sidebar.slider("SVM-RBF: C", 0.01, 10.0, 1.0, 0.01)
elif clf_name == "Logistic Regression":
    logreg_C = st.sidebar.slider("LogReg: C", 0.01, 10.0, 1.0, 0.01)
elif clf_name == "Random Forest":
    rf_n = st.sidebar.slider("RF: n_estimators", 50, 500, 300, 50)
    rf_depth = st.sidebar.slider("RF: max_depth (None=0)", 0, 50, 0, 1)

st.sidebar.divider()
st.sidebar.caption("缺失值将以列中位数填补；非树模型会做标准化。类别不平衡时大多数模型会启用 class_weight='balanced'（KNN除外）。")


# --- File upload ---
uploaded = st.file_uploader("📤 上传 Excel 文件（.xlsx）", type=["xlsx"])
if uploaded is None:
    st.info("请上传 Excel。第一列必须是**原始标签**（数字/字符串都可以），后面列是数值特征。")
    st.stop()

# Load excel
try:
    bytes_data = uploaded.read()
    xls = pd.ExcelFile(io.BytesIO(bytes_data))
    sheet_name = st.selectbox("选择工作表", xls.sheet_names)
    df = pd.read_excel(io.BytesIO(bytes_data), sheet_name=sheet_name)
except Exception as e:
    st.error(f"读取 Excel 失败：{e}")
    st.stop()

if df.shape[1] < 2:
    st.error("至少需要两列：第一列为原始标签，其余列为特征。")
    st.stop()

st.subheader("数据预览")
st.dataframe(df.head(), use_container_width=True)

# Identify columns
orig_label_col = df.columns[0]
feature_cols_all = [c for c in df.columns[1:]]

# Numeric casting for features; non-numeric will be coerced to NaN
df_features = df[feature_cols_all].apply(pd.to_numeric, errors="coerce")

# Show NaN report
with st.expander("🔎 缺失值报告"):
    na_counts = df_features.isna().sum()
    st.write(na_counts.to_frame("NaN计数"))
    st.caption("特征列中的非数值、空白会被转为 NaN；稍后会使用中位数填补。")

# --- Target class mapping ---
st.subheader("🎯 标签映射与分类设置")
unique_orig_labels = pd.Series(df[orig_label_col].unique()).tolist()
unique_orig_labels_sorted = sorted(unique_orig_labels, key=lambda x: str(x))

num_classes = st.radio("目标分类数", [2, 3, 4, 5], horizontal=True)

st.write("请为每个**原始标签值**指定一个新的**目标类别**编号（0..K-1）：")
mapping: Dict = {}
cols = st.columns(2)
for idx, val in enumerate(unique_orig_labels_sorted):
    with cols[idx % 2]:
        new_lab = st.selectbox(
            f"原始标签 {val} →",
            options=list(range(num_classes)),
            key=f"map_{idx}",
            index=0
        )
        mapping[val] = new_lab

st.caption(f"当前映射：{mapping}")

# Build y_new using mapping
try:
    y_new = df[orig_label_col].map(mapping)
except Exception as e:
    st.error(f"标签映射失败：{e}")
    st.stop()

# Drop rows where mapping failed (shouldn't happen if mapping is defined for all)
mask_valid = y_new.notna()
dropped = (~mask_valid).sum()
if dropped > 0:
    st.warning(f"有 {dropped} 行标签未成功映射，已移除。请检查映射是否覆盖全部原始标签。")
y = y_new[mask_valid].astype(int).values
X_all = df_features.loc[mask_valid].copy()

# Feature selection
st.subheader("🧩 特征选择")
default_feats = feature_cols_all  # 默认全选
chosen_feats = st.multiselect(
    "选择用于训练的特征列（可多选）",
    options=feature_cols_all,
    default=default_feats
)

if len(chosen_feats) == 0:
    st.error("请至少选择一个特征。")
    st.stop()

X = X_all[chosen_feats].values

# Train/test split (stratified)
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
except ValueError as e:
    st.error(f"划分数据失败：{e}（可能某一类样本太少，无法分层划分）")
    st.stop()

# Build model pipeline
num_classes_int = int(num_classes)
average_mode = "binary" if num_classes_int == 2 else "macro"

def make_non_tree_pipeline(base_estimator):
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if use_pca and clf_name != "Random Forest":
        steps.append(("pca", PCA(n_components=pca_var, svd_solver="full")))
    steps.append(("clf", base_estimator))
    return Pipeline(steps)

def make_tree_pipeline(base_estimator):
    # 树模型不需要标准化；但保留填补
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", base_estimator)
    ])

# Instantiate classifier
if clf_name == "KNN":
    clf = make_non_tree_pipeline(KNeighborsClassifier(n_neighbors=knn_k))
elif clf_name == "Linear SVM":
    # LinearSVC 不支持 predict_proba；class_weight 平衡
    clf = make_non_tree_pipeline(LinearSVC(C=linsvm_C, class_weight="balanced", dual="auto"))
elif clf_name == "SVM (RBF)":
    clf = make_non_tree_pipeline(SVC(C=rbf_C, kernel="rbf", class_weight="balanced", probability=True))
elif clf_name == "Logistic Regression":
    clf = make_non_tree_pipeline(LogisticRegression(C=logreg_C, max_iter=2000, class_weight="balanced"))
elif clf_name == "Random Forest":
    md = None if rf_depth == 0 else rf_depth
    clf = make_tree_pipeline(RandomForestClassifier(
        n_estimators=rf_n,
        max_depth=md,
        random_state=random_state,
        class_weight="balanced"
    ))
else:
    st.error("未知的分类器")
    st.stop()

# Train
st.subheader("🚀 训练与评估")
if st.button("训练并评估", type="primary"):
    try:
        clf.fit(X_train, y_train)
    except Exception as e:
        st.error(f"训练失败：{e}")
        st.stop()

    # Predict
    y_pred = clf.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=average_mode)

    left, right, right2 = st.columns(3)
    with left:
        st.metric("Accuracy", f"{acc:.4f}")
    with right:
        st.metric(f"F1 ({average_mode})", f"{f1:.4f}")

    # ROC AUC only for binary if proba/decision available
    auc_shown = False
    if num_classes_int == 2:
        try:
            # decision_function preferred; else predict_proba
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
                auc_shown = True
        except Exception:
            pass
    if num_classes_int != 2:
        st.caption("多分类暂不计算 AUC（可扩展成 OVO/OVR）。")

    # Confusion matrix
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
    # write numbers
    for i in range(num_classes_int):
        for j in range(num_classes_int):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    st.pyplot(fig)

    # Classification report (text)
    st.subheader("📄 分类报告")
    cr = classification_report(y_test, y_pred, digits=4, zero_division=0)
    st.code(cr, language="text")

    # Feature importance / coefficients
    st.subheader("🧭 可解释性")
    try:
        if clf_name == "Random Forest":
            # Extract inner estimator
            final = clf.named_steps["clf"]
            importances = final.feature_importances_
            # If PCA applied (we disabled for RF), skip mapping
            imp_df = pd.DataFrame({"feature": chosen_feats, "importance": importances}).sort_values("importance", ascending=False)
            st.dataframe(imp_df, use_container_width=True)
            fig2, ax2 = plt.subplots(figsize=(6, min(0.35*len(chosen_feats)+1, 10)))
            ax2.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
            ax2.set_title("Feature Importance (RF)")
            st.pyplot(fig2)

        elif clf_name in ["Linear SVM", "Logistic Regression"]:
            # pipeline may include scaler (and maybe PCA). 如果用了 PCA，系数在降维空间，解释性弱
            final = clf.named_steps["clf"]
            if hasattr(final, "coef_"):
                coef = final.coef_
                if num_classes_int == 2:
                    coef = coef[0]
                    names = chosen_feats
                    if use_pca:
                        st.warning("使用了 PCA：线性模型系数在降维后的特征空间，直接对应原始特征的解释性较弱。")
                    coef_df = pd.DataFrame({"feature": names, "coef": coef if len(coef)==len(names) else np.nan})
                    coef_df = coef_df.sort_values("coef", key=np.abs, ascending=False)
                    st.dataframe(coef_df, use_container_width=True)
                    fig3, ax3 = plt.subplots(figsize=(6, min(0.35*len(chosen_feats)+1, 10)))
                    ax3.barh(coef_df["feature"][::-1], coef_df["coef"][::-1])
                    ax3.set_title("Linear Model Coefficients")
                    st.pyplot(fig3)
                else:
                    st.caption("多分类线性系数为 one-vs-rest 矩阵，暂不绘制，可在数据框中查看。")
                    st.dataframe(pd.DataFrame(coef, columns=chosen_feats), use_container_width=True)
            else:
                st.caption("该模型没有可用的线性系数。")
        else:
            st.caption("KNN / SVM-RBF 没有直接的特征重要性。")

    except Exception as e:
        st.warning(f"可解释性计算遇到问题：{e}")

st.divider()
st.caption("小贴士：如果样本太少导致某类在测试集中为 0，试着减小测试集比例或合并类别。")
