import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AgroSense AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  GLOBAL CSS  – dark agri-tech aesthetic
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #0a0f0d; color: #e8f5e9; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1a12 0%, #0a0f0d 100%);
    border-right: 1px solid #1e3a26;
}
section[data-testid="stSidebar"] * { color: #c8e6c9 !important; }
.stSlider > div > div > div > div { background: #2e7d32 !important; }

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #0d1a12 0%, #122318 100%);
    border: 1px solid #1e4d2b;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    transition: border-color .3s;
}
.metric-card:hover { border-color: #4caf50; }
.metric-label { font-size: 11px; letter-spacing: 2px; color: #66bb6a; text-transform: uppercase; }
.metric-value { font-size: 28px; font-weight: 700; color: #e8f5e9; margin: 4px 0; }
.metric-unit  { font-size: 11px; color: #558b2f; }

/* ── Pump card ── */
.pump-on  { border: 2px solid #00e676; background: #001a08; box-shadow: 0 0 24px #00e67630; }
.pump-off { border: 2px solid #ef5350; background: #1a0000; box-shadow: 0 0 24px #ef535030; }
.pump-title { font-size: 13px; letter-spacing: 3px; color: #81c784; text-transform: uppercase; }
.pump-status-on  { font-size: 42px; font-weight: 700; color: #00e676; }
.pump-status-off { font-size: 42px; font-weight: 700; color: #ef5350; }
.pump-reason { font-size: 12px; color: #a5d6a7; margin-top: 6px; font-family: 'JetBrains Mono', monospace; }

/* ── Action badge ── */
.action-badge {
    display: inline-block;
    padding: 10px 22px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 15px;
    letter-spacing: 0.5px;
    margin: 8px 0;
}
.badge-irrigate    { background: #0d47a1; color: #90caf9; border: 1px solid #1565c0; }
.badge-fertilizer  { background: #1b5e20; color: #a5d6a7; border: 1px solid #2e7d32; }
.badge-pesticide   { background: #4a148c; color: #ce93d8; border: 1px solid #6a1b9a; }
.badge-monitor     { background: #e65100; color: #ffcc80; border: 1px solid #bf360c; }

/* ── Confidence bar wrapper ── */
.conf-wrap { background: #122318; border-radius: 8px; height: 8px; overflow: hidden; }
.conf-bar  { height: 8px; border-radius: 8px;
             background: linear-gradient(90deg, #2e7d32, #66bb6a, #00e676); }

/* ── History table ── */
.dataframe { background: #0d1a12 !important; color: #e8f5e9 !important; }

/* ── Section headers ── */
.section-head {
    font-size: 11px; letter-spacing: 3px; color: #4caf50;
    text-transform: uppercase; border-bottom: 1px solid #1e3a26;
    padding-bottom: 8px; margin: 24px 0 16px;
}

/* ── Model badge ── */
.model-chip {
    display: inline-block; background: #1e3a26; border: 1px solid #2e7d32;
    border-radius: 6px; padding: 4px 12px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #a5d6a7;
}

/* ── Alert-style boxes ── */
.alert-warn { background: #1a1200; border: 1px solid #f9a825; border-radius: 10px;
              padding: 12px 16px; color: #ffe082; font-size: 13px; }
.alert-info { background: #001220; border: 1px solid #1565c0; border-radius: 10px;
              padding: 12px 16px; color: #90caf9; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    model      = joblib.load("model.pkl")
    scaler     = joblib.load("scaler.pkl")
    le         = joblib.load("label_encoder.pkl")
    model_name = joblib.load("model_name.pkl")
    results    = joblib.load("model_results.pkl")

    pump_model = None
    if os.path.exists("pump_model.pkl"):
        pump_model = joblib.load("pump_model.pkl")

    features = ['N','P','K','Moisture','pH','Temperature','Humidity']
    if os.path.exists("features.pkl"):
        features = joblib.load("features.pkl")

    return model, scaler, le, model_name, results, pump_model, features

model, scaler, le, model_name, results, pump_model, FEATURES = load_artifacts()

# ══════════════════════════════════════════════════════════════
#  PUMP DECISION  (ML first, fallback to rule)
# ══════════════════════════════════════════════════════════════
def decide_pump(scaled_input, raw: dict):
    if pump_model is not None:
        pump_pred = pump_model.predict(scaled_input)[0]
        pump_prob = pump_model.predict_proba(scaled_input)[0][1]

        if raw["moisture"] < 20:
            reason = "⚠️ Critical moisture deficit"
        elif raw["temperature"] > 38 and raw["humidity"] < 35:
            reason = "🌡️ Heat + dryness stress"
        elif pump_pred == 1:
            reason = f"🤖 ML model → irrigate ({pump_prob*100:.0f}% confidence)"
        else:
            reason = f"🤖 ML model → no irrigation ({(1-pump_prob)*100:.0f}% confidence)"

        on = bool(pump_pred == 1)
        return on, pump_prob if on else 1 - pump_prob, reason

    # Rule-based fallback
    if raw["moisture"] < 20:
        return True, 0.95, "⚠️ Critical moisture deficit"
    if raw["temperature"] > 35 and raw["humidity"] < 40:
        return True, 0.80, "🌡️ Heat + dryness stress"
    return False, 0.85, "✅ Soil conditions adequate"

# ══════════════════════════════════════════════════════════════
#  ACTION METADATA
# ══════════════════════════════════════════════════════════════
ACTION_META = {
    "Irrigate":          {"icon": "💧", "badge": "badge-irrigate",   "tip": "Activate drip or sprinkler system now."},
    "Apply Fertilizer":  {"icon": "🧪", "badge": "badge-fertilizer", "tip": "Apply NPK blend per deficiency reading."},
    "Apply Pesticide":   {"icon": "🐛", "badge": "badge-pesticide",  "tip": "Spray recommended pesticide at dawn/dusk."},
    "Monitor":           {"icon": "🌿", "badge": "badge-monitor",    "tip": "No immediate action; continue monitoring."},
}

# ══════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state.history = []

# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style='padding:28px 0 12px'>
  <div style='font-size:11px;letter-spacing:4px;color:#4caf50;text-transform:uppercase;'>AI-Powered</div>
  <div style='font-size:38px;font-weight:700;line-height:1.1;color:#e8f5e9;'>
    AgroSense <span style='color:#00e676;'>Intelligence</span>
  </div>
  <div style='font-size:13px;color:#558b2f;margin-top:6px;'>
    Smart irrigation & crop management decision support
  </div>
</div>
""", unsafe_allow_html=True)

# Model chip
st.markdown(f"<span class='model-chip'>🤖 {model_name}</span>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  SIDEBAR – SENSOR INPUTS
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📡 Sensor Inputs")
    st.markdown("Adjust the sliders to match current field readings.")
    st.markdown("---")

    n  = st.slider("🌱 Nitrogen (N)",       0, 140, 50,  help="Soil nitrogen level (kg/ha)")
    p  = st.slider("🔵 Phosphorus (P)",     0, 140, 35,  help="Soil phosphorus level (kg/ha)")
    k  = st.slider("🟠 Potassium (K)",      0, 200, 70,  help="Soil potassium level (kg/ha)")
    m  = st.slider("💧 Moisture (%)",       8.0, 38.0, 20.0, step=0.5, help="Volumetric soil moisture content")
    ph = st.slider("⚗️ pH",                 4.0, 9.0, 6.5, step=0.1, help="Soil pH level")
    t  = st.slider("🌡️ Temperature (°C)",  18.0, 42.0, 30.0, step=0.5, help="Ambient air temperature")
    h  = st.slider("💨 Humidity (%)",       10.0, 100.0, 60.0, step=1.0, help="Relative humidity")

    st.markdown("---")
    predict_btn = st.button("🚀  Run AI Analysis", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Model Comparison")
    if results:
        for mname, info in results.items():
            if isinstance(info, dict):
                acc_val = info.get("test_accuracy", info)
            else:
                acc_val = info
            bar_pct = int(float(acc_val) * 100)
            st.markdown(f"""
            <div style='margin-bottom:10px'>
              <div style='font-size:11px;color:#81c784;'>{mname}</div>
              <div class='conf-wrap'>
                <div class='conf-bar' style='width:{bar_pct}%'></div>
              </div>
              <div style='font-size:11px;color:#558b2f;text-align:right;'>{bar_pct}%</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  SENSOR OVERVIEW CARDS  (always visible)
# ══════════════════════════════════════════════════════════════
st.markdown("<div class='section-head'>Current Sensor Readings</div>", unsafe_allow_html=True)

sensor_data = [
    ("N",           n,  "kg/ha",  "🌱"),
    ("P",           p,  "kg/ha",  "🔵"),
    ("K",           k,  "kg/ha",  "🟠"),
    ("Moisture",    m,  "%",      "💧"),
    ("pH",          ph, "",       "⚗️"),
    ("Temperature", t,  "°C",    "🌡️"),
    ("Humidity",    h,  "%",      "💨"),
]

cols = st.columns(7)
for col, (label, val, unit, icon) in zip(cols, sensor_data):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
          <div style='font-size:20px;'>{icon}</div>
          <div class='metric-label'>{label}</div>
          <div class='metric-value'>{val:.1f}</div>
          <div class='metric-unit'>{unit}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  CHART ROW
# ══════════════════════════════════════════════════════════════
st.markdown("<div class='section-head'>Sensor Profile</div>", unsafe_allow_html=True)

plt.style.use("dark_background")
fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
fig.patch.set_facecolor("#0a0f0d")

# Bar chart
ax1 = axes[0]
vals   = [n, p, k, m, ph, t, h]
labels = ["N", "P", "K", "Moisture", "pH", "Temp", "Humidity"]
colors = ["#43a047","#00897b","#558b2f","#0288d1","#fdd835","#e53935","#8e24aa"]

bars = ax1.bar(labels, vals, color=colors, edgecolor="#1e3a26", linewidth=0.8)
ax1.set_facecolor("#0d1a12")
ax1.tick_params(colors="#81c784", labelsize=9)
ax1.spines[:].set_color("#1e3a26")
ax1.set_title("Current Readings", color="#c8e6c9", fontsize=11, pad=8)
for bar, v in zip(bars, vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{v:.1f}", ha="center", va="bottom", fontsize=8, color="#c8e6c9")

# Radar / spider chart
ax2 = axes[1]
norm_vals = [
    n/140, p/140, k/200, m/38, ph/9, t/42, h/100
]
theta = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
norm_vals_c = norm_vals + [norm_vals[0]]
theta_c     = np.append(theta, theta[0])

ax2 = plt.subplot(1, 2, 2, polar=True, facecolor="#0d1a12")
ax2.plot(theta_c, norm_vals_c, color="#00e676", linewidth=1.8)
ax2.fill(theta_c, norm_vals_c, alpha=0.18, color="#00e676")
ax2.set_xticks(theta)
ax2.set_xticklabels(labels, color="#81c784", fontsize=8)
ax2.set_yticklabels([])
ax2.spines["polar"].set_color("#1e3a26")
ax2.set_facecolor("#0d1a12")
ax2.grid(color="#1e3a26", linewidth=0.5)
ax2.set_title("Normalized Profile", color="#c8e6c9", fontsize=11, pad=14)

fig.tight_layout(pad=1.5)
st.pyplot(fig)
plt.close(fig)

# ══════════════════════════════════════════════════════════════
#  PREDICTION BLOCK
# ══════════════════════════════════════════════════════════════
if predict_btn:
    raw_dict = dict(nitrogen=n, phosphorus=p, potassium=k,
                    moisture=m, ph=ph, temperature=t, humidity=h)

    # Compute engineered features (must match train_model.py exactly)
    water_stress   = (t * (100 - h)) / (m + 1)
    nutrient_score = n + p + k
    ph_deviation   = abs(ph - 6.5)

    # Build full feature vector — 10 columns if engineered features saved, else 7
    BASE_VALS = [n, p, k, m, ph, t, h]
    if len(FEATURES) == 10:
        all_vals = BASE_VALS + [water_stress, nutrient_score, ph_deviation]
    else:
        all_vals = BASE_VALS

    input_df = pd.DataFrame([all_vals], columns=FEATURES)
    scaled   = scaler.transform(input_df)

    # Action prediction
    pred_idx  = model.predict(scaled)[0]
    action    = le.inverse_transform([pred_idx])[0]
    conf      = float(np.max(model.predict_proba(scaled))) * 100
    meta      = ACTION_META.get(action, {"icon": "🌾", "badge": "badge-monitor", "tip": ""})

    # Pump decision
    pump_on, pump_conf, pump_reason = decide_pump(scaled, raw_dict)

    st.markdown("<div class='section-head'>AI Analysis Results</div>", unsafe_allow_html=True)

    left, mid, right = st.columns([1, 1, 1.3])

    # ── Pump card ──
    with left:
        pump_class  = "pump-on"  if pump_on else "pump-off"
        pump_label  = "PUMP ON"  if pump_on else "PUMP OFF"
        status_cls  = "pump-status-on" if pump_on else "pump-status-off"
        water_line  = "💧💧💧 Water flowing" if pump_on else "🚫 No water flow"
        st.markdown(f"""
        <div class='metric-card {pump_class}' style='padding:22px;'>
          <div class='pump-title'>Irrigation Pump</div>
          <div class='{status_cls}'>{pump_label}</div>
          <div style='font-size:13px;color:#e8f5e9;margin:6px 0;'>{water_line}</div>
          <div class='pump-reason'>{pump_reason}</div>
          <div style='margin-top:10px;font-size:11px;color:#558b2f;'>
            Pump confidence: {pump_conf*100:.0f}%
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Action card ──
    with mid:
        st.markdown(f"""
        <div class='metric-card' style='padding:22px;'>
          <div class='pump-title'>Recommended Action</div>
          <div style='font-size:38px;margin:6px 0;'>{meta['icon']}</div>
          <div>
            <span class='action-badge {meta["badge"]}'>{action}</span>
          </div>
          <div style='margin-top:10px;font-size:12px;color:#81c784;'>{meta['tip']}</div>
        </div>""", unsafe_allow_html=True)

    # ── Confidence card ──
    with right:
        bar_w = int(conf)
        st.markdown(f"""
        <div class='metric-card' style='padding:22px;'>
          <div class='pump-title'>Model Confidence</div>
          <div class='metric-value' style='font-size:40px;'>{conf:.1f}%</div>
          <div style='margin:10px 0;'>
            <div class='conf-wrap'>
              <div class='conf-bar' style='width:{bar_w}%'></div>
            </div>
          </div>
          <div style='font-size:11px;color:#558b2f;'>Model: {model_name}</div>
        </div>""", unsafe_allow_html=True)

        # Stress alerts
        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        if m < 20:
            st.markdown("<div class='alert-warn'>⚠️ Moisture critically low — irrigation urgent</div>",
                        unsafe_allow_html=True)
        if t > 38:
            st.markdown("<div class='alert-warn'>🌡️ Temperature extreme — monitor crop stress</div>",
                        unsafe_allow_html=True)
        if ph < 5.5:
            st.markdown("<div class='alert-warn'>⚗️ pH too acidic — consider liming</div>",
                        unsafe_allow_html=True)
        if ph > 8:
            st.markdown("<div class='alert-warn'>⚗️ pH too alkaline — consider acidifying agents</div>",
                        unsafe_allow_html=True)
        if n < 15:
            st.markdown("<div class='alert-info'>🌱 Low nitrogen — foliar spray recommended</div>",
                        unsafe_allow_html=True)

    # ── Class probability chart ──
    proba = model.predict_proba(scaled)[0]
    classes = le.classes_

    fig2, ax = plt.subplots(figsize=(7, 2.6))
    fig2.patch.set_facecolor("#0a0f0d")
    ax.set_facecolor("#0d1a12")

    bar_colors = ["#00e676" if c == action else "#2e7d32" for c in classes]
    hbars = ax.barh(classes, proba * 100, color=bar_colors, edgecolor="#1e3a26")
    ax.set_xlabel("Confidence (%)", color="#81c784", fontsize=9)
    ax.tick_params(colors="#81c784", labelsize=9)
    ax.spines[:].set_color("#1e3a26")
    ax.set_xlim(0, 100)
    for bar, v in zip(hbars, proba * 100):
        ax.text(v + 0.5, bar.get_y() + bar.get_height()/2,
                f"{v:.1f}%", va="center", fontsize=8, color="#c8e6c9")
    ax.set_title("Action Probability Distribution", color="#c8e6c9", fontsize=10, pad=8)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # ── Save to history ──
    st.session_state.history.append({
        "N": n, "P": p, "K": k,
        "Moisture": m, "pH": ph,
        "Temp (°C)": t, "Humidity": h,
        "Action": action,
        "Confidence": f"{conf:.1f}%",
        "Pump": "ON 💧" if pump_on else "OFF 🚫",
    })

# ══════════════════════════════════════════════════════════════
#  PREDICTION HISTORY
# ══════════════════════════════════════════════════════════════
if st.session_state.history:
    st.markdown("<div class='section-head'>Prediction History</div>", unsafe_allow_html=True)

    df_hist = pd.DataFrame(st.session_state.history)
    st.dataframe(df_hist, use_container_width=True, hide_index=True)

    col_dl, col_clr = st.columns([3, 1])
    with col_dl:
        csv = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Report as CSV", csv,
                           file_name="agrosense_report.csv", mime="text/csv")
    with col_clr:
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<hr style='border-color:#1e3a26;margin-top:40px;'>
<p style='text-align:center;color:#2e7d32;font-size:12px;letter-spacing:2px;'>
  AGROSENSE INTELLIGENCE · AI-POWERED PRECISION AGRICULTURE · v2.0
</p>
""", unsafe_allow_html=True)