import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
df = pd.read_csv("Crop_Recommendation.csv")

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
# Step 3: Streamlit UI
# -------------------------------
st.set_page_config(page_title="Crop Recommendation", page_icon="🌱", layout="centered")
st.title("🌱 Smart Crop Recommendation System")

col1, col2 = st.columns(2)

with col1:
    nitrogen = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
    phosphorus = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
    potassium = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
    ph = st.slider("Soil pH", 0.0, 14.0, 6.5)

with col2:
    temperature = st.number_input("Temperature (°C)", min_value=0, max_value=50, value=25)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 70.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0, max_value=500, value=150)

# -------------------------------
# Step 4: Prediction
# -------------------------------
if st.button("🔍 Recommend Crop"):
    sample = [[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]]
    probs = model.predict_proba(sample)[0]

    # Sort top 3 recommendations
    top_idx = probs.argsort()[::-1][:3]
    top_crops = [(crop_enc.inverse_transform([i])[0], probs[i]*100) for i in top_idx]

    # Highlight best crop
    best_crop, best_conf = top_crops[0]

    # Build recommendations HTML
    rec_html = "".join([
        f"<li style='margin: 8px 0; font-size: 18px;'><b>{c}</b> - <span style='color:green;'>{p:.2f}%</span></li>"
        for c, p in top_crops
    ])

    # -------------------------------
    # Custom Stylish Popup HTML
    # -------------------------------
    popup_html = f"""
    <style>
    @keyframes fadeIn {{
        from {{opacity: 0; transform: scale(0.9);}}
        to {{opacity: 1; transform: scale(1);}}
    }}
    #popup {{
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: rgba(0,0,0,0.6);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
        animation: fadeIn 0.4s ease;
    }}
    .popup-content {{
        background: linear-gradient(135deg, #ffffff 0%, #f0fff0 100%);
        padding: 25px;
        border-radius: 16px;
        width: 420px;
        text-align: center;
        box-shadow: 0px 6px 20px rgba(0,0,0,0.25);
        font-family: 'Segoe UI', sans-serif;
        animation: fadeIn 0.4s ease;
    }}
    .popup-content h2 {{
        margin-top: 0;
        color: #2e7d32;
    }}
    .best-crop {{
        background: #e8f5e9;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 15px;
        font-size: 20px;
        font-weight: bold;
        color: #1b5e20;
    }}
    button {{
        padding: 10px 20px;
        border: none;
        background: #2e7d32;
        color: white;
        border-radius: 8px;
        cursor: pointer;
        margin-top: 10px;
        font-size: 16px;
    }}
    button:hover {{
        background: #1b5e20;
    }}
    </style>

    <div id="popup">
        <div class="popup-content">
            <h2>🌱 Recommended Crops</h2>
            <div class="best-crop">✅ Best Choice: {best_crop} ({best_conf:.2f}%)</div>
            <ul style="list-style:none; padding:0; text-align:left;">
                {rec_html}
            </ul>
            <button onclick="document.getElementById('popup').style.display='none'">Close</button>
        </div>
    </div>
    """
    st.components.v1.html(popup_html, height=400)

# -------------------------------
# Step 5: Show Accuracy
# -------------------------------
st.markdown("### Model Accuracy")
y_pred = model.predict(X)
acc = (y_pred == y).mean()
st.metric("📊 Training Data Accuracy", f"{acc*100:.2f}%")
