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
# Step 3: Streamlit UI Config
# -------------------------------
st.set_page_config(page_title="Smart Crop Recommender", page_icon="🌱", layout="wide")

st.markdown("""
    <style>
    body { background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%); }
    .main { background: transparent; }
    .stNumberInput, .stSlider { background: white; border-radius: 12px; padding: 10px; }
    h1 {
        text-align: center;
        color: #1b5e20;
        font-size: 42px;
        font-family: 'Segoe UI', sans-serif;
        margin-bottom: 20px;
    }
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
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        nitrogen = st.number_input("🌿 Nitrogen (N)", min_value=0, max_value=200, value=50)
        phosphorus = st.number_input("🌱 Phosphorus (P)", min_value=0, max_value=200, value=50)
        potassium = st.number_input("🌻 Potassium (K)", min_value=0, max_value=200, value=50)
        ph = st.slider("🧪 Soil pH", 0.0, 14.0, 6.5)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        temperature = st.number_input("🌡️ Temperature (°C)", min_value=0, max_value=50, value=25)
        humidity = st.slider("💧 Humidity (%)", 0.0, 100.0, 70.0)
        rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0, max_value=500, value=150)
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Step 5: Prediction Button
# -------------------------------
if st.button("🔍 Recommend Crops", use_container_width=True):
    sample = [[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]]
    probs = model.predict_proba(sample)[0]

    # Sort top 3 recommendations
    top_idx = probs.argsort()[::-1][:3]
    top_crops = [(crop_enc.inverse_transform([i])[0], probs[i]*100) for i in top_idx]

    best_crop, best_conf = top_crops[0]

    # Add crop details if available in CSV
    crop_details = {}
    if "Details" in df.columns:
        for c in top_crops:
            name = c[0]
            detail_row = df[df["Crop"] == name].iloc[0]
            crop_details[name] = detail_row["Details"]
    else:
        for c in top_crops:
            crop_details[c[0]] = "No details available in dataset."

    # Build recommendations HTML
    rec_html = "".join([
        f"""
        <li style='margin: 12px 0; font-size: 18px;'>
            <b>{c}</b> - <span style='color:green;'>{p:.2f}%</span><br>
            <small style='color:#555;'>{crop_details[c]}</small>
        </li>
        """
        for c, p in top_crops
    ])

    # -------------------------------
    # Stylish Popup with Details
    # -------------------------------
    popup_html = f"""
    <style>
    @keyframes fadeIn {{
        from {{opacity: 0; transform: translateY(-20px);}}
        to {{opacity: 1; transform: translateY(0);}}
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
        background: #ffffff;
        padding: 30px;
        border-radius: 20px;
        width: 500px;
        text-align: center;
        box-shadow: 0px 10px 25px rgba(0,0,0,0.25);
        font-family: 'Segoe UI', sans-serif;
        animation: fadeIn 0.4s ease;
    }}
    .popup-content h2 {{
        margin-top: 0;
        color: #2e7d32;
        font-size: 26px;
    }}
    .best-crop {{
        background: linear-gradient(90deg, #81c784, #388e3c);
        padding: 12px;
        border-radius: 12px;
        margin-bottom: 15px;
        font-size: 20px;
        font-weight: bold;
        color: white;
    }}
    button {{
        padding: 10px 25px;
        border: none;
        background: #2e7d32;
        color: white;
        border-radius: 10px;
        cursor: pointer;
        margin-top: 15px;
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
    st.components.v1.html(popup_html, height=500)

# -------------------------------
# Step 6: Show Accuracy
# -------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 📊 Model Accuracy")
y_pred = model.predict(X)
acc = (y_pred == y).mean()
st.metric("Training Accuracy", f"{acc*100:.2f}%")
st.markdown("</div>", unsafe_allow_html=True)
