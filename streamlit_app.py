"""
🍷 Wine Varietal Classifier — Full-featured Streamlit App
- Train & compare 6 ML models
- Predict wine varietal from user inputs
- Explore dataset interactively
- View performance metrics & charts
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🍷 Wine Classifier",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #7b2d8b;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        color: white;
    }
    .metric-card h3 { font-size: 0.85rem; color: #c084fc; margin: 0 0 4px 0; }
    .metric-card h2 { font-size: 1.6rem; margin: 0; color: white; }
    .winner-badge {
        background: linear-gradient(90deg, #7b2d8b, #a855f7);
        color: white; border-radius: 20px;
        padding: 2px 12px; font-size: 0.75rem; font-weight: bold;
    }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); }
    [data-testid="stSidebar"] * { color: #f1f5f9 !important; }
    [data-testid="stSidebar"] .stRadio label p { color: #f1f5f9 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { color: #f1f5f9 !important; }
    [data-testid="stSidebar"] hr { border-color: #4a1d6e; }
    h1, h2, h3 { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["Class 1 (Cultivar A)", "Class 2 (Cultivar B)", "Class 3 (Cultivar C)"]
CLASS_COLORS = ["#a855f7", "#ec4899", "#f59e0b"]
MODEL_COLORS = ["#6366f1", "#ec4899", "#10b981", "#f59e0b", "#3b82f6", "#ef4444"]

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(random_state=42),
    "KNN":                 KNeighborsClassifier(),
    "Naive Bayes":         GaussianNB(),
    "SVM":                 SVC(probability=True, random_state=42),
}

# ── Data & model training ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train():
    data    = load_wine()
    X       = pd.DataFrame(data.data, columns=data.feature_names)
    y       = pd.Series(data.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    trained, metrics_list, cv_scores_all = {}, [], {}

    for name, clf in MODELS.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        cv   = cross_val_score(pipe, X, y, cv=10, scoring="accuracy")
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)

        trained[name]        = pipe
        cv_scores_all[name]  = cv
        metrics_list.append({
            "Model":     name,
            "Accuracy":  round(acc,  4),
            "Precision": round(prec, 4),
            "Recall":    round(rec,  4),
            "F1-Score":  round(f1,   4),
            "CV Mean":   round(cv.mean(), 4),
            "CV Std":    round(cv.std(),  4),
            "CM":        cm,
        })

    metrics_df = pd.DataFrame([{k: v for k, v in m.items() if k != "CM"} for m in metrics_list])
    return X, y, X_train, X_test, y_train, y_test, trained, metrics_df, metrics_list, cv_scores_all, data.feature_names

with st.spinner("🍷 Training 6 models with 10-fold cross-validation…"):
    (X, y, X_train, X_test, y_train, y_test,
     trained_models, metrics_df, metrics_list,
     cv_scores_all, feature_names) = load_and_train()

best_model_name = metrics_df.loc[metrics_df["Accuracy"].idxmax(), "Model"]

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍷 Wine Classifier")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🔮 Predict Varietal",
        "📊 Model Comparison",
        "📈 Performance Charts",
        "🔍 Explore Dataset",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"**Dataset:** {len(X)} samples · {len(feature_names)} features")
    st.markdown(f"**Classes:** 3 wine cultivars")
    st.markdown(f"**Best Model:** {best_model_name}")
    st.markdown(f"**Best Accuracy:** {metrics_df['Accuracy'].max():.1%}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT VARIETAL
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predict Varietal":
    st.title("🔮 Predict Wine Varietal")
    st.markdown("Adjust the physicochemical properties below and see all 6 models predict the wine class simultaneously.")
    st.divider()

    # Feature inputs in a 3-column grid
    feat_defaults = X.mean().to_dict()
    feat_mins     = X.min().to_dict()
    feat_maxs     = X.max().to_dict()

    user_vals = {}
    cols = st.columns(3)
    for i, feat in enumerate(feature_names):
        with cols[i % 3]:
            user_vals[feat] = st.slider(
                feat.replace("_", " ").title(),
                min_value=float(round(feat_mins[feat], 2)),
                max_value=float(round(feat_maxs[feat], 2)),
                value=float(round(feat_defaults[feat], 2)),
                step=float(round((feat_maxs[feat] - feat_mins[feat]) / 100, 3)),
            )

    st.divider()

    if st.button("🔍 Predict with All 6 Models", type="primary", use_container_width=True):
        input_df = pd.DataFrame([user_vals])
        st.markdown("### 🏆 Predictions")

        pred_cols = st.columns(3)
        for i, (name, pipe) in enumerate(trained_models.items()):
            pred       = pipe.predict(input_df)[0]
            proba      = pipe.predict_proba(input_df)[0]
            confidence = proba.max()
            is_best    = name == best_model_name

            with pred_cols[i % 3]:
                badge = " 🏆" if is_best else ""
                st.markdown(f"**{name}{badge}**")
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<h3>Predicted Class</h3>"
                    f"<h2>{CLASS_NAMES[pred]}</h2>"
                    f"<p style='color:#c084fc;font-size:0.8rem'>Confidence: {confidence:.1%}</p>"
                    f"</div>", unsafe_allow_html=True
                )

                # Mini probability bar chart
                fig, ax = plt.subplots(figsize=(3, 1.2))
                fig.patch.set_facecolor("#1a1a2e")
                ax.set_facecolor("#1a1a2e")
                bars = ax.barh(["C1","C2","C3"], proba, color=CLASS_COLORS, height=0.5)
                ax.set_xlim(0, 1)
                ax.tick_params(colors="white", labelsize=7)
                for spine in ax.spines.values(): spine.set_visible(False)
                ax.xaxis.set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close()
                st.markdown("")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    st.markdown("Side-by-side metrics for all 6 models trained on the Wine dataset with 10-fold cross-validation.")
    st.divider()

    # Highlight best per column
    display_df = metrics_df.set_index("Model")
    def highlight_max(s):
        is_max = s == s.max()
        return ["background-color: #4a1d6e; color: white; font-weight: bold" if v else "" for v in is_max]

    st.dataframe(
        display_df.style.apply(highlight_max, subset=["Accuracy","Precision","Recall","F1-Score","CV Mean"]).format("{:.4f}"),
        use_container_width=True, height=260
    )

    st.divider()

    # Confusion matrices for all 6
    st.markdown("### Confusion Matrices")
    cm_cols = st.columns(3)
    for i, m in enumerate(metrics_list):
        with cm_cols[i % 3]:
            st.markdown(f"**{m['Model']}**")
            fig, ax = plt.subplots(figsize=(3.5, 2.8))
            fig.patch.set_facecolor("#1a1a2e")
            ax.set_facecolor("#1a1a2e")
            sns.heatmap(
                m["CM"], annot=True, fmt="d", cmap="Purples",
                xticklabels=["C1","C2","C3"], yticklabels=["C1","C2","C3"],
                ax=ax, linewidths=0.5, linecolor="#2d2d4e",
                annot_kws={"color":"white","size":11}
            )
            ax.tick_params(colors="white", labelsize=8)
            ax.set_xlabel("Predicted", color="white", fontsize=8)
            ax.set_ylabel("Actual", color="white", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PERFORMANCE CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Performance Charts":
    st.title("📈 Performance Charts")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Metric Comparison", "CV Score Distribution", "Feature Importance"])

    # ── Tab 1: Bar chart of all metrics ──────────────────────────────────────
    with tab1:
        metric_choice = st.selectbox("Metric", ["Accuracy","Precision","Recall","F1-Score","CV Mean"])
        vals  = metrics_df[metric_choice].values
        names = metrics_df["Model"].values
        order = np.argsort(vals)[::-1]

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        bars = ax.bar(
            [names[i] for i in order],
            [vals[i] for i in order],
            color=[MODEL_COLORS[i % 6] for i in order],
            edgecolor="#2d2d4e", linewidth=0.8
        )
        for bar, val in zip(bars, [vals[i] for i in order]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=9)
        ax.set_ylim(min(vals) * 0.97, 1.01)
        ax.set_ylabel(metric_choice, color="white")
        ax.tick_params(colors="white", axis="both")
        ax.tick_params(axis="x", rotation=20)
        for spine in ax.spines.values(): spine.set_color("#2d2d4e")
        ax.yaxis.grid(True, color="#2d2d4e", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Tab 2: CV box plots ───────────────────────────────────────────────────
    with tab2:
        st.markdown("10-fold cross-validation score distribution for each model.")
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        data_to_plot = [cv_scores_all[m] for m in MODELS.keys()]
        bp = ax.boxplot(data_to_plot, patch_artist=True, medianprops={"color":"white","linewidth":2})
        for patch, color in zip(bp["boxes"], MODEL_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for element in ["whiskers","caps","fliers"]:
            for item in bp[element]: item.set_color("#aaaaaa")
        ax.set_xticklabels(list(MODELS.keys()), rotation=20, ha="right", color="white")
        ax.tick_params(colors="white")
        ax.set_ylabel("CV Accuracy", color="white")
        for spine in ax.spines.values(): spine.set_color("#2d2d4e")
        ax.yaxis.grid(True, color="#2d2d4e", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Tab 3: Feature importance (Random Forest) ─────────────────────────────
    with tab3:
        st.markdown("Feature importances from the Random Forest model.")
        rf_pipe    = trained_models["Random Forest"]
        importances = rf_pipe.named_steps["clf"].feature_importances_
        feat_order  = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        bars = ax.barh(
            [feature_names[i] for i in feat_order[::-1]],
            importances[feat_order[::-1]],
            color="#a855f7", edgecolor="#2d2d4e"
        )
        ax.tick_params(colors="white", labelsize=9)
        ax.set_xlabel("Importance", color="white")
        for spine in ax.spines.values(): spine.set_color("#2d2d4e")
        ax.xaxis.grid(True, color="#2d2d4e", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EXPLORE DATASET
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Explore Dataset":
    st.title("🔍 Explore the Wine Dataset")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Data Table", "Distributions", "Correlation Matrix"])

    # ── Tab 1: Raw data ───────────────────────────────────────────────────────
    with tab1:
        display_X        = X.copy()
        display_X["Class"] = y.map({0: "Cultivar A", 1: "Cultivar B", 2: "Cultivar C"})
        class_filter = st.multiselect("Filter by class", ["Cultivar A","Cultivar B","Cultivar C"],
                                       default=["Cultivar A","Cultivar B","Cultivar C"])
        filtered = display_X[display_X["Class"].isin(class_filter)]
        st.dataframe(filtered, use_container_width=True, height=400)
        st.caption(f"Showing {len(filtered)} of {len(X)} samples")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples", len(X))
        col2.metric("Features", len(feature_names))
        col3.metric("Cultivar A", int((y == 0).sum()))
        col4.metric("Cultivar B / C", f"{int((y==1).sum())} / {int((y==2).sum())}")

    # ── Tab 2: Feature distributions ─────────────────────────────────────────
    with tab2:
        feat_select = st.selectbox("Select feature", feature_names)
        fig, ax     = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        for cls_idx, (cls_name, color) in enumerate(zip(["Cultivar A","Cultivar B","Cultivar C"], CLASS_COLORS)):
            vals = X.loc[y == cls_idx, feat_select]
            ax.hist(vals, bins=20, alpha=0.6, color=color, label=cls_name, edgecolor="#1a1a2e")
        ax.set_xlabel(feat_select, color="white")
        ax.set_ylabel("Count", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values(): spine.set_color("#2d2d4e")
        ax.yaxis.grid(True, color="#2d2d4e", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        legend = ax.legend(facecolor="#2d2d4e", labelcolor="white", edgecolor="#4a1d6e")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Summary stats
        st.markdown("**Summary Statistics by Class**")
        stats = pd.concat([
            X.loc[y == i, feat_select].describe().rename(f"Cultivar {chr(65+i)}")
            for i in range(3)
        ], axis=1).round(3)
        st.dataframe(stats, use_container_width=True)

    # ── Tab 3: Correlation matrix ─────────────────────────────────────────────
    with tab3:
        corr = X.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="RdPu",
            center=0, ax=ax, linewidths=0.5, linecolor="#2d2d4e",
            annot_kws={"size": 7, "color": "white"},
            cbar_kws={"shrink": 0.8}
        )
        ax.tick_params(colors="white", labelsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()