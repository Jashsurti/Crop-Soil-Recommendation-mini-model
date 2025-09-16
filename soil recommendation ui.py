import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
url = "https://github.com/Jashsurti/Crop-Soil-Recommendation-mini-model/blob/main/Crop_Recommendation_With_Soil.csv"
df = pd.read_csv(url)

# Encode Crop Labels
crop_enc = LabelEncoder()
df['CropLabel'] = crop_enc.fit_transform(df['Crop'])

# Features and Target
X = df.drop(columns=['Crop', 'CropLabel'])
y = df['CropLabel']

# -------------------------------
# Step 2: Train Model
# -------------------------------
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X, y)

# -------------------------------
# Step 3: Streamlit UI Config
# -------------------------------
st.set_page_config(page_title="Smart Crop Recommender", page_icon="🌱", layout="wide")

st.markdown("""
    <style>
    body { background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%); }
    h1 { text-align: center; color: #1b5e20; font-size: 42px; }
    .card {
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.1);
        margin: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🌾 Smart Crop Recommendation System</h1>", unsafe_allow_html=True)

# -------------------------------
# Step 4: Input Fields
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    nitrogen = st.number_input("🌿 Nitrogen (N)", min_value=0, max_value=200, value=50)
    phosphorus = st.number_input("🌱 Phosphorus (P)", min_value=0, max_value=200, value=50)
    potassium = st.number_input("🌻 Potassium (K)", min_value=0, max_value=200, value=50)
    ph = st.slider("🧪 Soil pH", 0.0, 14.0, 6.5)

with col2:
    temperature = st.number_input("🌡️ Temperature (°C)", min_value=0, max_value=50, value=25)
    humidity = st.slider("💧 Humidity (%)", 0.0, 100.0, 70.0)
    rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0, max_value=500, value=150)

# -------------------------------
# Step 5: Prediction
# -------------------------------
if st.button("🔍 Recommend Crop"):
    sample = [[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]]
    probs = model.predict_proba(sample)[0]

    # Get top 3 crops
    top_indices = probs.argsort()[-3:][::-1]
    top_crops = [(crop_enc.inverse_transform([i])[0], probs[i]*100) for i in top_indices]

    # -------------------------------
    # Custom Popup HTML (Clean, No Bar Chart)
    # -------------------------------
    recommendations_html = "".join(
        [f"<p style='font-size:16px;'>✅ <b>{crop}</b> — {conf:.2f}%</p>" for crop, conf in top_crops]
    )

    popup_html = f"""
    <div id="popup" style="
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: rgba(0,0,0,0.6);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;">
        <div style="
            background: white;
            padding: 20px;
            border-radius: 12px;
            width: 380px;
            text-align: center;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.3);">
            
            <h2 style="color:#2e7d32;">🌱 Recommended Crops</h2>
            {recommendations_html}
            
            <button onclick="document.getElementById('popup').style.display='none'"
                style="padding: 8px 16px; border: none; background: #2e7d32; 
                       color: white; border-radius: 8px; cursor: pointer; margin-top:10px;">
                Close
            </button>
        </div>
    </div>
    """
    st.components.v1.html(popup_html, height=300)

# -------------------------------
# Step 6: Show Accuracy
# -------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 📊 Model Accuracy")
y_pred = model.predict(X)
acc = (y_pred == y).mean()
st.metric("Training Accuracy", f"{acc*100:.2f}%")
st.markdown("</div>", unsafe_allow_html=True)
