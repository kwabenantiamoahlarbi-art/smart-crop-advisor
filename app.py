import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import plotly.express as px

# ──────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
#  DARK THEME CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
  /* ---- Base & Background ---- */
  html, body, [data-testid="stAppViewContainer"] {
      background-color: #0d1117 !important;
      color: #e6edf3 !important;
  }
  [data-testid="stSidebar"] {
      background-color: #161b22 !important;
      border-right: 1px solid #30363d;
  }
  [data-testid="stHeader"] { background-color: #0d1117 !important; }

  /* ---- Cards ---- */
  .card {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 18px;
  }
  .result-card {
      background: linear-gradient(135deg, #0f2c1a 0%, #1a3a2a 100%);
      border: 1px solid #2ea043;
      border-radius: 16px;
      padding: 32px;
      text-align: center;
      margin-bottom: 18px;
  }
  .tips-card {
      background: #1c2128;
      border: 1px solid #388bfd44;
      border-radius: 12px;
      padding: 22px;
      margin-bottom: 18px;
  }

  /* ---- Typography ---- */
  h1 { color: #58a6ff !important; letter-spacing: -0.5px; }
  h2, h3 { color: #79c0ff !important; }
  .crop-name {
      font-size: 3rem;
      font-weight: 800;
      color: #3fb950;
      text-transform: capitalize;
      letter-spacing: 1px;
  }
  .success-msg {
      font-size: 1.1rem;
      color: #7ee787;
      margin-top: 8px;
  }
  .subtitle {
      color: #8b949e;
      font-size: 1rem;
      margin-top: -12px;
      margin-bottom: 28px;
  }
  .section-label {
      font-size: 0.78rem;
      font-weight: 700;
      color: #58a6ff;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 4px;
  }
  .tip-item {
      padding: 6px 0;
      color: #cdd9e5;
      font-size: 0.95rem;
      border-bottom: 1px solid #21262d;
  }
  .badge {
      display: inline-block;
      padding: 3px 12px;
      border-radius: 20px;
      font-size: 0.78rem;
      font-weight: 600;
      background: #1f6feb33;
      color: #58a6ff;
      border: 1px solid #1f6feb;
      margin: 2px;
  }
  .accuracy-pill {
      background: #1a3a2a;
      border: 1px solid #2ea043;
      color: #3fb950;
      padding: 4px 14px;
      border-radius: 20px;
      font-size: 0.82rem;
      font-weight: 700;
  }
  /* ---- Slider & Input overrides ---- */
  [data-testid="stSlider"] > div > div { color: #e6edf3 !important; }
  .stSlider [class*="thumb"] { background: #58a6ff !important; }
  .stSlider [class*="track"] { background: #30363d !important; }
  div[data-baseweb="select"] > div { background: #21262d !important; border-color: #30363d !important; }

  /* ---- Button ---- */
  .stButton > button {
      background: linear-gradient(90deg, #238636, #2ea043) !important;
      color: white !important;
      border: none !important;
      border-radius: 10px !important;
      font-size: 1.05rem !important;
      font-weight: 700 !important;
      padding: 14px 0 !important;
      width: 100% !important;
      letter-spacing: 0.5px;
      transition: all 0.2s;
  }
  .stButton > button:hover {
      background: linear-gradient(90deg, #2ea043, #3fb950) !important;
      transform: translateY(-1px);
      box-shadow: 0 4px 16px #2ea04355;
  }
  /* ---- Divider ---- */
  hr { border-color: #21262d !important; }
  /* ---- Metric ---- */
  [data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 1.6rem !important; }
  [data-testid="stMetricLabel"] { color: #8b949e !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  CROP TIPS DATABASE
# ──────────────────────────────────────────────
CROP_TIPS = {
    "rice": {
        "emoji": "🌾", "season": "Kharif (Jun–Nov)",
        "soil": "Clay or loamy soil with good water retention. Ideal pH: 6.0–7.0.",
        "climate": "Requires high humidity (~80%) and warm temperatures (20–27°C).",
        "fertilizer": "Apply urea in splits; add phosphorus before sowing.",
        "water": "Needs consistent flooding (5–10 cm water depth) during growing period.",
        "tips": ["Transplant seedlings at 3–4 weeks old.", "Control weeds during early growth.", "Drain fields 2 weeks before harvest."]
    },
    "maize": {
        "emoji": "🌽", "season": "Kharif & Rabi",
        "soil": "Well-drained loamy soil. pH: 5.8–7.0.",
        "climate": "Warm weather (18–27°C) with moderate rainfall (60–110 mm).",
        "fertilizer": "High nitrogen demand; apply NPK 120:60:40 kg/ha.",
        "water": "Moderate; critical at tasseling and silking stages.",
        "tips": ["Plant in rows for mechanized harvesting.", "Intercrop with legumes to improve soil.", "Watch for stem borer insects."]
    },
    "chickpea": {
        "emoji": "🫘", "season": "Rabi (Oct–Mar)",
        "soil": "Sandy loam to clay loam; well-drained. pH: 6.0–8.0.",
        "climate": "Cool and dry (15–25°C). Low humidity tolerance.",
        "fertilizer": "Low N; use starter dose + phosphorus 40 kg/ha.",
        "water": "Drought-tolerant; irrigation at pre-flowering and pod-filling.",
        "tips": ["Inoculate seeds with Rhizobium for N-fixation.", "Avoid waterlogging.", "Harvest when leaves turn yellow."]
    },
    "kidneybeans": {
        "emoji": "🫘", "season": "Kharif",
        "soil": "Well-drained loamy or sandy loam. pH: 6.0–7.0.",
        "climate": "Mild weather (18–24°C); sensitive to frost.",
        "fertilizer": "Moderate N-P-K; Rhizobium inoculation recommended.",
        "water": "Regular irrigation; avoid waterlogging.",
        "tips": ["Stake plants for better yield.", "Pick pods before full maturity.", "Rotate with cereals to reduce disease."]
    },
    "pigeonpeas": {
        "emoji": "🫘", "season": "Kharif",
        "soil": "Sandy loam to clay; moderate drainage. pH: 5.0–7.0.",
        "climate": "Tropical/subtropical; tolerates 25–35°C and moderate humidity.",
        "fertilizer": "Minimal N; 20:50:20 NPK at sowing.",
        "water": "Drought-tolerant; one irrigation at flowering if rain fails.",
        "tips": ["Plant in mixed cropping with sorghum or maize.", "Prune after first harvest for ratoon crop.", "Controls soil erosion effectively."]
    },
    "mothbeans": {
        "emoji": "🫘", "season": "Kharif",
        "soil": "Sandy loam; excellent drainage. pH: 6.5–7.5.",
        "climate": "Hot and dry (25–35°C); low rainfall (40–60 mm).",
        "fertilizer": "Low requirements; 10:25:0 NPK sufficient.",
        "water": "Minimal; extremely drought-resistant.",
        "tips": ["Ideal for arid zones.", "Fixes nitrogen naturally.", "Good fodder crop after harvest."]
    },
    "mungbean": {
        "emoji": "🫘", "season": "Kharif/Zaid",
        "soil": "Well-drained loamy soil. pH: 6.2–7.2.",
        "climate": "Warm (25–35°C) with moderate humidity (~80%).",
        "fertilizer": "Starter dose only; N-fixation covers requirements.",
        "water": "Light irrigation; sensitive to waterlogging.",
        "tips": ["Short duration crop (60–70 days).", "Good as green manure.", "Harvest pods as they ripen to avoid shattering."]
    },
    "blackgram": {
        "emoji": "🫘", "season": "Kharif & Rabi",
        "soil": "Loamy to clay loam; good drainage. pH: 6.0–7.5.",
        "climate": "Hot and humid (25–35°C) conditions.",
        "fertilizer": "20:40:20 NPK; Rhizobium inoculation helpful.",
        "water": "Moderate; critical at flowering and pod development.",
        "tips": ["Sow at proper spacing (30×10 cm).", "Spray for thrips and aphid control.", "High protein crop — excellent food value."]
    },
    "lentil": {
        "emoji": "🫘", "season": "Rabi",
        "soil": "Sandy loam to clay; pH: 6.0–8.0.",
        "climate": "Cool (15–25°C); tolerates mild frost at seedling stage.",
        "fertilizer": "20:40:20 NPK; responds well to phosphorus.",
        "water": "One or two irrigations; at branching and pod-filling.",
        "tips": ["Pre-soak seeds to improve germination.", "Weed control in first 30 days is critical.", "Excellent rotation crop with wheat."]
    },
    "pomegranate": {
        "emoji": "🍈", "season": "Perennial",
        "soil": "Deep loamy soil; tolerates saline conditions. pH: 5.5–7.5.",
        "climate": "Semi-arid (20–35°C); tolerates drought; high humidity at fruiting.",
        "fertilizer": "NPK 625:125:375 g/plant/year.",
        "water": "Drip irrigation preferred; consistent moisture at fruiting.",
        "tips": ["Prune for open canopy structure.", "Watch for bacterial blight disease.", "Fruit cracking prevented by even watering."]
    },
    "banana": {
        "emoji": "🍌", "season": "Year-round",
        "soil": "Rich, well-drained loamy soil. pH: 6.0–7.5.",
        "climate": "Tropical (25–30°C); high humidity (~80%); no frost.",
        "fertilizer": "High K demand; 200:60:300 NPK g/plant.",
        "water": "High water needs (1200–2200 mm); drip irrigation ideal.",
        "tips": ["Remove suckers regularly; keep one ratoon.", "Support pseudostem with bamboo stake.", "Harvest when fingers fill out and turn light green."]
    },
    "mango": {
        "emoji": "🥭", "season": "Summer (Mar–Jun)",
        "soil": "Deep alluvial/loamy; well-drained. pH: 5.5–7.5.",
        "climate": "Hot-dry flowering period (30–35°C); tropical/subtropical.",
        "fertilizer": "NPK 1:0.5:1 ratio; increase K before fruiting.",
        "water": "Minimal during flowering; regular irrigation at fruit development.",
        "tips": ["Prune after harvest for next season.", "Protect flowers from rain and frost.", "Mango malformation controlled by pruning infected parts."]
    },
    "grapes": {
        "emoji": "🍇", "season": "Rabi (Oct–Mar harvest)",
        "soil": "Well-drained sandy loam or gravelly loam. pH: 6.5–7.0.",
        "climate": "Warm, dry summers; cool winters; low humidity during ripening.",
        "fertilizer": "High P and K; 200:150:200 NPK g/vine.",
        "water": "Drip irrigation; reduce before harvest for sugar concentration.",
        "tips": ["Train on trellis/bower system.", "Bagging clusters prevents bird damage.", "Thin clusters for larger berry size."]
    },
    "watermelon": {
        "emoji": "🍉", "season": "Summer (Feb–Jun)",
        "soil": "Sandy loam; well-drained; pH: 6.0–7.0.",
        "climate": "Hot and sunny (25–35°C); high humidity during growing.",
        "fertilizer": "High N at vegetative stage; shift to K and P at fruiting.",
        "water": "Consistent moisture; reduce 1–2 weeks before harvest for sweetness.",
        "tips": ["Pollination by bees is critical — avoid pesticides during flowering.", "Tap the fruit — hollow sound means ripe.", "Mulch to retain soil moisture."]
    },
    "muskmelon": {
        "emoji": "🍈", "season": "Summer (Mar–Jun)",
        "soil": "Sandy loam; well-drained. pH: 6.0–7.0.",
        "climate": "Very warm (28–35°C); high humidity (~90%).",
        "fertilizer": "Moderate N-P-K; increase K for sweetness.",
        "water": "Moderate; reduce irrigation as fruit matures.",
        "tips": ["Grow on raised beds for drainage.", "Slip test — ripe fruit detaches easily from vine.", "Cover fruits with nets to prevent sunburn."]
    },
    "apple": {
        "emoji": "🍎", "season": "Temperate summer",
        "soil": "Deep, well-drained loamy or silty loam. pH: 5.5–6.5.",
        "climate": "Cool (22–24°C); needs chilling hours in winter (below 7°C).",
        "fertilizer": "Balanced NPK; calcium sprays for fruit quality.",
        "water": "Drip or sprinkler; avoid waterlogging.",
        "tips": ["Thin fruits for larger size.", "Cross-pollination with 2+ varieties improves yield.", "Protect from woolly aphid and scab disease."]
    },
    "orange": {
        "emoji": "🍊", "season": "Winter (Nov–Mar)",
        "soil": "Deep sandy loam; well-drained. pH: 6.0–7.5.",
        "climate": "Subtropical (22–32°C); high humidity (~90%).",
        "fertilizer": "High N requirement; NPK 400:200:400 g/tree.",
        "water": "Regular; critical at fruit development; drip recommended.",
        "tips": ["Weed control in young orchards is essential.", "Watch for citrus greening disease (HLB).", "Harvest by color change and juice content."]
    },
    "papaya": {
        "emoji": "🍈", "season": "Year-round",
        "soil": "Well-drained, fertile loamy soil. pH: 6.5–7.0.",
        "climate": "Tropical (25–35°C); very high humidity (~92%). Frost-sensitive.",
        "fertilizer": "NPK 250:250:500 g/plant; monthly application.",
        "water": "Regular light irrigation; never waterlog.",
        "tips": ["Plant 1 male for every 10 female plants.", "Fast-maturing — fruit in 9–11 months.", "Remove diseased leaves promptly."]
    },
    "coconut": {
        "emoji": "🥥", "season": "Perennial",
        "soil": "Sandy loam; coastal alluvium preferred. pH: 5.2–8.0.",
        "climate": "Tropical coastal (27°C avg); very high humidity (~95%).",
        "fertilizer": "NPK 500:320:1200 g/palm/year.",
        "water": "Regular; 600–2000 mm rainfall or equivalent irrigation.",
        "tips": ["Intercrop with banana, cocoa, or vegetables.", "Takes 6–10 years to first produce.", "Harvest every 45 days once mature."]
    },
    "cotton": {
        "emoji": "🌿", "season": "Kharif (Apr–Nov)",
        "soil": "Deep black cotton soil (regur). pH: 6.0–8.0.",
        "climate": "Hot (21–30°C); low humidity; long frost-free season.",
        "fertilizer": "High N; NPK 120:60:60 kg/ha.",
        "water": "Moderate; critical at square, flower, and boll formation stages.",
        "tips": ["Use Bt cotton varieties for bollworm resistance.", "Topping at 8–10 nodes increases boll set.", "Timely harvest prevents fiber quality loss."]
    },
    "jute": {
        "emoji": "🌿", "season": "Kharif (Mar–Jun)",
        "soil": "Alluvial loamy; pH: 6.0–7.0.",
        "climate": "Warm and humid (25–35°C); high rainfall (150–200 mm).",
        "fertilizer": "NPK 40:20:20 kg/ha.",
        "water": "Needs adequate moisture; suited to flood-prone areas.",
        "tips": ["Harvest before flowering for best fiber quality.", "Ret stalks in slow-moving water for fiber extraction.", "Good rotation crop with paddy."]
    },
    "coffee": {
        "emoji": "☕", "season": "Perennial",
        "soil": "Deep, well-drained red laterite or forest loam. pH: 6.0–6.5.",
        "climate": "Subtropical shade conditions (20–28°C); moderate humidity (~60%).",
        "fertilizer": "NPK 75:75:75 g/plant + organic manure.",
        "water": "Moderate and well-distributed; critical at post-monsoon.",
        "tips": ["Grow under shade trees (Silver Oak recommended).", "Prune for 3-tier canopy management.", "Harvest only ripe red cherries for quality."]
    },
}

# ──────────────────────────────────────────────
#  MODEL TRAINING (cached)
# ──────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_csv("crop_data.csv")
    X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]].values
    y = df["label"].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, le, acc

model, label_encoder, model_accuracy = train_model()

# ──────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 About the System")
    st.markdown("""
    <div style='color:#8b949e; font-size:0.92rem; line-height:1.7'>
    This system uses a <strong style='color:#58a6ff'>Gaussian Naïve Bayes</strong>
    classifier trained on 2,200 real agricultural records across <strong style='color:#58a6ff'>22 crops</strong>.<br><br>
    Enter your soil and climate measurements to receive an intelligent crop recommendation.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.metric("Algorithm", "Gaussian Naïve Bayes")
    st.metric("Training Samples", "1,760")
    st.metric("Test Accuracy", f"{model_accuracy * 100:.1f}%")

    st.markdown("---")
    st.markdown("### 🌍 Supported Crops")
    crops_grid = ""
    for crop, info in CROP_TIPS.items():
        crops_grid += f"<span class='badge'>{info['emoji']} {crop.title()}</span>"
    st.markdown(crops_grid, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='color:#484f58; font-size:0.78rem; text-align:center'>
    Smart Crop Recommendation System v1.0<br>
    Powered by Gaussian Naïve Bayes ML
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────
col_icon, col_title = st.columns([0.06, 0.94])
with col_icon:
    st.markdown("<div style='font-size:3rem; margin-top:8px'>🌾</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("# Smart Crop Recommendation System")
    st.markdown("<p class='subtitle'>AI-powered advisory tool — enter your soil and climate data to discover the best crop for your field</p>", unsafe_allow_html=True)

st.markdown("---")

# ──────────────────────────────────────────────
#  INSTRUCTIONS BANNER
# ──────────────────────────────────────────────
st.markdown("""
<div class='card'>
  <p class='section-label'>📋 How to Use</p>
  <p style='color:#cdd9e5; margin:0; line-height:1.8'>
  <strong style='color:#79c0ff'>Step 1</strong> — Use the sliders below to enter your <strong>soil nutrient levels</strong> (N, P, K) and <strong>soil pH</strong>.<br>
  <strong style='color:#79c0ff'>Step 2</strong> — Set the <strong>climate parameters</strong> for your region: temperature, humidity, and annual rainfall.<br>
  <strong style='color:#79c0ff'>Step 3</strong> — Click <strong style='color:#3fb950'>Predict Best Crop</strong> and the AI will recommend the optimal crop with confidence scores and expert tips.
  </p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  INPUT SECTIONS
# ──────────────────────────────────────────────
st.markdown("### 🧪 Soil Nutrient Levels")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<p class='section-label'>🌱 Nitrogen (N) — kg/ha</p>", unsafe_allow_html=True)
    N = st.slider("Nitrogen", 0, 140, 60, label_visibility="collapsed")
    st.markdown(f"<center style='color:#58a6ff; font-size:1.3rem; font-weight:700'>{N}</center>", unsafe_allow_html=True)

with col2:
    st.markdown("<p class='section-label'>🔵 Phosphorus (P) — kg/ha</p>", unsafe_allow_html=True)
    P = st.slider("Phosphorus", 5, 145, 53, label_visibility="collapsed")
    st.markdown(f"<center style='color:#58a6ff; font-size:1.3rem; font-weight:700'>{P}</center>", unsafe_allow_html=True)

with col3:
    st.markdown("<p class='section-label'>🟡 Potassium (K) — kg/ha</p>", unsafe_allow_html=True)
    K = st.slider("Potassium", 5, 205, 48, label_visibility="collapsed")
    st.markdown(f"<center style='color:#58a6ff; font-size:1.3rem; font-weight:700'>{K}</center>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🌡️ Climate Parameters")
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("<p class='section-label'>🌡️ Temperature — °C</p>", unsafe_allow_html=True)
    temperature = st.slider("Temperature", 8.0, 44.0, 25.0, step=0.1, label_visibility="collapsed")
    st.markdown(f"<center style='color:#f0883e; font-size:1.3rem; font-weight:700'>{temperature:.1f}°C</center>", unsafe_allow_html=True)

with col5:
    st.markdown("<p class='section-label'>💧 Humidity — %</p>", unsafe_allow_html=True)
    humidity = st.slider("Humidity", 14.0, 100.0, 71.0, step=0.1, label_visibility="collapsed")
    st.markdown(f"<center style='color:#58a6ff; font-size:1.3rem; font-weight:700'>{humidity:.1f}%</center>", unsafe_allow_html=True)

with col6:
    st.markdown("<p class='section-label'>🌧️ Rainfall — mm</p>", unsafe_allow_html=True)
    rainfall = st.slider("Rainfall", 20.0, 299.0, 103.0, step=0.1, label_visibility="collapsed")
    st.markdown(f"<center style='color:#58a6ff; font-size:1.3rem; font-weight:700'>{rainfall:.1f} mm</center>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🧬 Soil Chemistry")
col7, col_space = st.columns([0.4, 0.6])

with col7:
    st.markdown("<p class='section-label'>⚗️ Soil pH Level (0–14)</p>", unsafe_allow_html=True)
    ph = st.slider("Soil pH", 3.5, 10.0, 6.5, step=0.01, label_visibility="collapsed")
    ph_label = "Strongly Acidic" if ph < 5 else "Acidic" if ph < 6 else "Slightly Acidic" if ph < 6.5 else "Neutral" if ph < 7.5 else "Alkaline"
    ph_color = "#f85149" if ph < 5 else "#f0883e" if ph < 6 else "#d29922" if ph < 7 else "#3fb950" if ph < 7.5 else "#388bfd"
    st.markdown(f"<center style='color:{ph_color}; font-size:1.3rem; font-weight:700'>{ph:.2f} — {ph_label}</center>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  PREDICT BUTTON
# ──────────────────────────────────────────────
col_btn1, col_btn2, col_btn3 = st.columns([0.25, 0.5, 0.25])
with col_btn2:
    predict_clicked = st.button("🌾 Predict Best Crop", use_container_width=True)

# ──────────────────────────────────────────────
#  PREDICTION OUTPUT
# ──────────────────────────────────────────────
if predict_clicked:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    proba = model.predict_proba(features)[0]
    pred_idx = np.argmax(proba)
    pred_crop = label_encoder.inverse_transform([pred_idx])[0]
    confidence = proba[pred_idx] * 100

    # Top 5 predictions
    top5_idx = np.argsort(proba)[::-1][:5]
    top5_crops = label_encoder.inverse_transform(top5_idx)
    top5_probs = proba[top5_idx] * 100

    tip_data = CROP_TIPS.get(pred_crop, {})
    emoji = tip_data.get("emoji", "🌿")

    # ── Success Card ──
    st.markdown(f"""
    <div class='result-card'>
      <div style='font-size:4rem'>{emoji}</div>
      <div class='crop-name'>{pred_crop.title()}</div>
      <div class='success-msg'>✅ This is the optimal crop for your soil and climate conditions</div>
      <div style='margin-top:14px'>
        <span class='accuracy-pill'>🎯 Confidence: {confidence:.1f}%</span>
        &nbsp;
        <span class='accuracy-pill'>📅 Season: {tip_data.get("season", "—")}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Two-column layout: Tips + Chart ──
    col_tips, col_chart = st.columns([0.48, 0.52])

    with col_tips:
        st.markdown(f"""
        <div class='tips-card'>
          <p class='section-label'>🌱 Expert Advisory — {pred_crop.title()}</p>

          <p style='color:#8b949e; font-size:0.78rem; margin:0 0 4px 0'>SOIL REQUIREMENT</p>
          <div class='tip-item'>{tip_data.get("soil", "—")}</div>

          <p style='color:#8b949e; font-size:0.78rem; margin:12px 0 4px 0'>CLIMATE CONDITIONS</p>
          <div class='tip-item'>{tip_data.get("climate", "—")}</div>

          <p style='color:#8b949e; font-size:0.78rem; margin:12px 0 4px 0'>FERTILIZER GUIDE</p>
          <div class='tip-item'>{tip_data.get("fertilizer", "—")}</div>

          <p style='color:#8b949e; font-size:0.78rem; margin:12px 0 4px 0'>WATER MANAGEMENT</p>
          <div class='tip-item'>{tip_data.get("water", "—")}</div>

          <p style='color:#8b949e; font-size:0.78rem; margin:12px 0 6px 0'>FARMING TIPS</p>
          {"".join([f'<div class="tip-item">• {t}</div>' for t in tip_data.get("tips", [])])}
        </div>
        """, unsafe_allow_html=True)

    with col_chart:
        # Confidence bar chart for top 5
        colors = ["#2ea043" if i == 0 else "#1f6feb" for i in range(5)]
        fig = go.Figure(go.Bar(
            x=top5_probs,
            y=[c.title() for c in top5_crops],
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="#30363d", width=1)
            ),
            text=[f"{p:.1f}%" for p in top5_probs],
            textposition="outside",
            textfont=dict(color="#e6edf3", size=13),
        ))
        fig.update_layout(
            title=dict(text="Top 5 Crop Recommendations", font=dict(color="#79c0ff", size=15)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#161b22",
            xaxis=dict(
                title="Confidence (%)",
                color="#8b949e",
                gridcolor="#21262d",
                showline=False,
                range=[0, max(top5_probs) * 1.25]
            ),
            yaxis=dict(color="#e6edf3", tickfont=dict(size=13)),
            margin=dict(l=10, r=40, t=50, b=30),
            height=310,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Gauge chart for top confidence
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            number={"suffix": "%", "font": {"color": "#3fb950", "size": 34}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                "bar": {"color": "#2ea043"},
                "bgcolor": "#21262d",
                "steps": [
                    {"range": [0, 40], "color": "#1c2128"},
                    {"range": [40, 70], "color": "#1c2128"},
                    {"range": [70, 100], "color": "#1c2128"},
                ],
                "threshold": {
                    "line": {"color": "#3fb950", "width": 4},
                    "thickness": 0.75,
                    "value": confidence
                }
            },
            title={"text": "Prediction Confidence", "font": {"color": "#79c0ff", "size": 13}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e6edf3"},
            margin=dict(l=20, r=20, t=40, b=10),
            height=220
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Input summary ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📋 Your Input Summary")
    summary_data = {
        "Parameter": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Temperature", "Humidity", "Soil pH", "Rainfall"],
        "Your Value": [f"{N} kg/ha", f"{P} kg/ha", f"{K} kg/ha", f"{temperature:.1f} °C", f"{humidity:.1f} %", f"{ph:.2f}", f"{rainfall:.1f} mm"],
        "Unit": ["kg/ha", "kg/ha", "kg/ha", "°C", "%", "pH", "mm/year"]
    }
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(
        df_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Parameter": st.column_config.TextColumn("Parameter"),
            "Your Value": st.column_config.TextColumn("Value Entered"),
            "Unit": st.column_config.TextColumn("Unit"),
        }
    )
