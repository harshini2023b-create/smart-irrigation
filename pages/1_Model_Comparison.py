import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #0a0f0d; color: #e8f5e9; }
.section-head {
    font-size: 11px; letter-spacing: 3px; color: #4caf50;
    text-transform: uppercase; border-bottom: 1px solid #1e3a26;
    padding-bottom: 8px; margin: 24px 0 16px;
}
.model-card {
    background: linear-gradient(135deg, #0d1a12, #122318);
    border: 1px solid #1e4d2b; border-radius: 12px;
    padding: 18px 22px; margin-bottom: 12px;
    transition: border-color .3s;
}
.model-card:hover { border-color: #4caf50; }
.model-name  { font-size: 14px; font-weight: 600; color: #e8f5e9; }
.model-acc   { font-size: 28px; font-weight: 700; color: #00e676; }
.model-cv    { font-size: 12px; color: #81c784; font-family: 'JetBrains Mono', monospace; }
.best-badge  { display:inline-block; background:#1b5e20; color:#a5d6a7;
               border:1px solid #2e7d32; border-radius:6px;
               padding:2px 10px; font-size:11px; margin-left:8px; }
.conf-wrap { background:#122318; border-radius:8px; height:10px; overflow:hidden; margin:8px 0 4px; }
.conf-bar  { height:10px; border-radius:8px;
             background: linear-gradient(90deg,#2e7d32,#66bb6a,#00e676); }
</style>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div style='padding:20px 0 10px'>
  <div style='font-size:11px;letter-spacing:4px;color:#4caf50;text-transform:uppercase;'>AgroSense AI</div>
  <div style='font-size:34px;font-weight:700;color:#e8f5e9;'>
    Advanced Model <span style='color:#00e676;'>Comparison</span>
  </div>
  <div style='font-size:13px;color:#558b2f;'>Performance benchmarks across all trained classifiers</div>
</div>
""", unsafe_allow_html=True)

# ── LOAD RESULTS ─────────────────────────────────────────────
try:
    results    = joblib.load("model_results.pkl")
    model_name = joblib.load("model_name.pkl")
except Exception as e:
    st.error(f"Could not load model results: {e}")
    st.stop()

# ── NORMALIZE: handle both float and dict stored results ──────
def extract_acc(val):
    """Accept float, int, or dict with test_accuracy / cv_accuracy keys."""
    if isinstance(val, dict):
        return (
            float(val.get("test_accuracy", val.get("accuracy", 0))),
            float(val.get("cv_accuracy",   val.get("cv",       0))),
        )
    return float(val), 0.0

rows = []
for name, val in results.items():
    test_acc, cv_acc = extract_acc(val)
    rows.append({
        "Model":         name,
        "Test Accuracy": test_acc,
        "CV Accuracy":   cv_acc,
        "Best":          name == model_name,
    })

# Sort by test accuracy (now guaranteed to be floats)
df = pd.DataFrame(rows).sort_values("Test Accuracy", ascending=False).reset_index(drop=True)

# ── METRIC SUMMARY CARDS ─────────────────────────────────────
st.markdown("<div class='section-head'>Model Leaderboard</div>", unsafe_allow_html=True)

cols = st.columns(len(df))
for col, (_, row) in zip(cols, df.iterrows()):
    badge = "<span class='best-badge'>★ Best</span>" if row["Best"] else ""
    bar_w = int(row["Test Accuracy"] * 100)
    cv_str = f"CV: {row['CV Accuracy']*100:.1f}%" if row["CV Accuracy"] > 0 else ""
    with col:
        st.markdown(f"""
        <div class='model-card'>
          <div class='model-name'>{row['Model']}{badge}</div>
          <div class='model-acc'>{row['Test Accuracy']*100:.1f}%</div>
          <div class='conf-wrap'><div class='conf-bar' style='width:{bar_w}%'></div></div>
          <div class='model-cv'>Test accuracy &nbsp;|&nbsp; {cv_str}</div>
        </div>""", unsafe_allow_html=True)

# ── BAR CHART ────────────────────────────────────────────────
st.markdown("<div class='section-head'>Accuracy Comparison Chart</div>", unsafe_allow_html=True)

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor("#0a0f0d")
ax.set_facecolor("#0d1a12")

bar_colors = ["#00e676" if n == model_name else "#2e7d32" for n in df["Model"]]
bars = ax.bar(df["Model"], df["Test Accuracy"] * 100, color=bar_colors,
              edgecolor="#1e3a26", linewidth=0.8, width=0.5)

# CV accuracy overlay dots
if df["CV Accuracy"].sum() > 0:
    ax.scatter(df["Model"], df["CV Accuracy"] * 100,
               color="#ffb300", zorder=5, s=60, label="CV Accuracy")
    ax.legend(facecolor="#0d1a12", edgecolor="#1e3a26", labelcolor="#e8f5e9", fontsize=9)

for bar, v in zip(bars, df["Test Accuracy"] * 100):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{v:.1f}%", ha="center", va="bottom", fontsize=9, color="#c8e6c9")

ax.set_ylabel("Accuracy (%)", color="#81c784", fontsize=10)
ax.set_ylim(0, 105)
ax.tick_params(colors="#81c784", labelsize=9)
ax.spines[:].set_color("#1e3a26")
ax.set_title("Model Test Accuracy (green bar) vs CV Accuracy (yellow dot)",
             color="#c8e6c9", fontsize=11, pad=10)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ── DATA TABLE ───────────────────────────────────────────────
st.markdown("<div class='section-head'>Full Results Table</div>", unsafe_allow_html=True)

display_df = df.copy()
display_df["Test Accuracy"] = (display_df["Test Accuracy"] * 100).round(2).astype(str) + "%"
display_df["CV Accuracy"]   = display_df["CV Accuracy"].apply(
    lambda x: f"{x*100:.2f}%" if x > 0 else "—"
)
display_df["Best"] = display_df["Best"].apply(lambda x: "★ Yes" if x else "")
display_df = display_df.rename(columns={"Best": "Selected"})

st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown(f"""
<div style='margin-top:16px;padding:14px 18px;background:#0d1a12;
            border:1px solid #1e4d2b;border-radius:10px;font-size:13px;color:#a5d6a7;'>
  ✅ &nbsp;<b>Best model selected:</b> {model_name} &nbsp;—&nbsp;
  chosen automatically based on highest test accuracy during training.
</div>
""", unsafe_allow_html=True)