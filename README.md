# 🌾 Smart Crop Recommendation System

An AI-powered agricultural advisory tool built with **Gaussian Naïve Bayes** ML and **Streamlit**.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Make sure `crop_data.csv` is in the same folder as `app.py`

Your project folder should look like:
```
smart-crop-advisor/
├── app.py
├── crop_data.csv
└── requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open automatically in your browser at `https://smart-crop-advisor-zx95sxmqgvuabswaegqktx.streamlit.app`

---

## 🧠 ML Model

| Property | Value |
|---|---|
| Algorithm | Gaussian Naïve Bayes |
| Training samples | 1,760 (80% split) |
| Test samples | 440 (20% split) |
| Features | N, P, K, Temperature, Humidity, pH, Rainfall |
| Output | 22 crop classes |

---

## 🌿 Features

- **7-parameter input** via interactive sliders (N, P, K, temperature, humidity, pH, rainfall)
- **Predicted crop name** with emoji and success message
- **Confidence score** (%) with gauge visualization
- **Top-5 crop ranking** horizontal bar chart
- **Expert advisory tips** per crop (soil, climate, fertilizer, water, farming tips)
- **Input summary table** for review
- **Dark theme** with high-contrast green accents
- **22 supported crops**: rice, maize, chickpea, kidney beans, pigeon peas, moth beans, mung bean, black gram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee

---

## 📦 Deploying to Streamlit Cloud (Free)

1. Push your project to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select the repo and set `app.py` as the main file
5. Click **Deploy** — done! 🎉

---

## 🛠️ Tech Stack

- **Frontend & Server**: Streamlit
- **ML Model**: scikit-learn (GaussianNB)
- **Data Processing**: pandas, numpy
- **Visualizations**: Plotly
