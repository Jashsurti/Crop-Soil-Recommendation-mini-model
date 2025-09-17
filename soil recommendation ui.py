import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# -------------------------------
# Step 1: Expanded Kerala Crop Dataset
# -------------------------------
data = {
    'Nitrogen': [
        90, 100, 85, 95,   # Rice
        30, 35, 40, 25,    # Banana
        45, 50, 55, 60,    # Pepper
        70, 75, 80, 85,    # Coconut
        60, 65, 55, 70,    # Mango
        40, 45, 50, 35,    # Tapioca
    ] * 2,
    'Phosphorus': [
        40, 45, 50, 42,
        60, 65, 55, 70,
        50, 55, 45, 60,
        65, 70, 60, 75,
        55, 60, 65, 50,
        45, 50, 55, 40,
    ] * 2,
    'Potassium': [
        40, 42, 38, 45,
        40, 45, 50, 42,
        60, 55, 65, 70,
        70, 75, 65, 80,
        55, 50, 60, 65,
        50, 45, 55, 40,
    ] * 2,
    'pH': [
        6.5, 6.2, 6.8, 6.4,
        5.5, 5.8, 5.6, 5.7,
        6.0, 6.1, 6.2, 5.9,
        6.8, 6.5, 6.6, 6.7,
        6.0, 6.2, 6.1, 5.9,
        5.8, 6.0, 6.1, 5.7,
    ] * 2,
    'Rainfall': [
        200, 220, 210, 230,
        100, 120, 90, 110,
        180, 170, 190, 160,
        300, 280, 320, 310,
        250, 240, 260, 270,
        150, 140, 160, 130,
    ] * 2,
    'Temperature': [
        28, 29, 27, 30,
        25, 26, 24, 27,
        27, 28, 26, 29,
        30, 31, 29, 32,
        26, 27, 28, 25,
        27, 26, 25, 28,
    ] * 2,
    'SoilType': [
        'Clay', 'Alluvial', 'Clay', 'Alluvial',
        'Loamy', 'Loamy', 'Loamy', 'Loamy',
        'Sandy', 'Loamy', 'Laterite', 'Sandy',
        'Laterite', 'Coastal', 'Laterite', 'Sandy',
        'Alluvial', 'Laterite', 'Alluvial', 'Clay',
        'Sandy', 'Loamy', 'Laterite', 'Clay',
    ] * 2,
    'Crop': [
        'Rice', 'Rice', 'Rice', 'Rice',
        'Banana', 'Banana', 'Banana', 'Banana',
        'Pepper', 'Pepper', 'Pepper', 'Pepper',
        'Coconut', 'Coconut', 'Coconut', 'Coconut',
        'Mango', 'Mango', 'Mango', 'Mango',
        'Tapioca', 'Tapioca', 'Tapioca', 'Tapioca',
    ] * 2
}

df = pd.DataFrame(data)

# -------------------------------
# Step 2: Encode Categorical
# -------------------------------
soil_enc = LabelEncoder()
crop_enc = LabelEncoder()

df['SoilTypeEnc'] = soil_enc.fit_transform(df['SoilType'])
df['CropLabel'] = crop_enc.fit_transform(df['Crop'])

X = df.drop(columns=['Crop', 'CropLabel', 'SoilType'])
y = df['CropLabel']

# -------------------------------
# Step 3: Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 4: Train Model
# -------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Crop Recommendation", page_icon="🌱", layout="centered")

st.title("🌱 Smart Crop Recommendation System - Kerala Edition")
st.write("Enter soil and weather details to get personalized crop suggestions for Kerala farmers.")

col1, col2 = st.columns(2)

with col1:
    nitrogen = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
    phosphorus = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
    potassium = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
    ph = st.slider("Soil pH", 0.0, 14.0, 6.0)

with col2:
    rainfall = st.number_input("Rainfall (mm)", min_value=0, max_value=500, value=120)
    temperature = st.number_input("Temperature (°C)", min_value=0, max_value=50, value=25)
    soil_type = st.selectbox("Soil Type", soil_enc.classes_)

# -------------------------------
# Crop Details & Economics
# -------------------------------
crop_details = {
    "Rice": "🌾 **Rice** grows best in clay or alluvial soil with high rainfall. Requires standing water and warm temperatures.",
    "Banana": "🍌 **Banana** thrives in loamy soil with moderate rainfall. Needs irrigation and rich organic matter.",
    "Pepper": "🌶️ **Pepper** prefers sandy or loamy soil with good drainage. Needs warm climate and partial shade.",
    "Coconut": "🥥 **Coconut** grows in sandy coastal or laterite soil. Requires high humidity and plenty of sunlight.",
    "Mango": "🥭 **Mango** prefers alluvial or laterite soil. Grows best in tropical climate with dry spells before flowering.",
    "Tapioca": "🍠 **Tapioca (Cassava)** is drought tolerant, grows in sandy-loam soil, and requires minimal rainfall."
}

crop_info = {
    "Rice": {"yield": "2500–3000 kg/acre", "price": "₹20–25/kg", "profit": "Medium"},
    "Banana": {"yield": "10,000–12,000 kg/acre", "price": "₹10–15/kg", "profit": "High"},
    "Pepper": {"yield": "400–600 kg/acre", "price": "₹350–450/kg", "profit": "Very High"},
    "Coconut": {"yield": "8000–10,000 nuts/acre", "price": "₹12–20/nut", "profit": "High"},
    "Mango": {"yield": "2000–2500 kg/acre", "price": "₹40–60/kg", "profit": "High"},
    "Tapioca": {"yield": "6000–8000 kg/acre", "price": "₹8–12/kg", "profit": "Medium"}
}

emoji_map = {
    "Rice": "🌾",
    "Banana": "🍌",
    "Pepper": "🌶️",
    "Coconut": "🥥",
    "Mango": "🥭",
    "Tapioca": "🍠"
}

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🔍 Recommend Crops"):
    sample = [[
        nitrogen, phosphorus, potassium, ph,
        rainfall, temperature, soil_enc.transform([soil_type])[0]
    ]]

    probs = model.predict_proba(sample)[0]

    top_n = 5
    top_indices = probs.argsort()[-top_n:][::-1]
    top_crops = [(crop_enc.inverse_transform([i])[0], probs[i]) for i in top_indices]

    # Popup-style container
    with st.container():
        st.markdown(
            """
            <div style="
                position:fixed; top:0; left:0; width:100%; height:100%;
                background:rgba(0,0,0,0.5); display:flex; justify-content:center; align-items:center;
                z-index:1000;">
              <div style="
                  background:white; padding:25px; border-radius:16px; width:65%;
                  box-shadow:0 4px 12px rgba(0,0,0,0.3); text-align:left;">
                  <h2>🌱 ✅ Recommendation Results</h2>
            """,
            unsafe_allow_html=True,
        )

        for crop, prob in top_crops:
            emoji = emoji_map.get(crop, "🌱")
            st.markdown(
                f"""
                <div style="padding:12px; border-radius:12px; background-color:#f6fdf6; 
                            margin-bottom:12px; box-shadow:0 1px 4px rgba(0,0,0,0.1);">
                    <h4 style="margin:0;">{emoji} {crop}</h4>
                    <p style="margin:0; color:gray;">Suitability: <b>{prob*100:.2f}%</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(float(prob))
            st.caption(crop_details.get(crop, "ℹ️ No details available."))

            info = crop_info.get(crop, {})
            if info:
                st.markdown(
                    f"**📈 Yield:** {info['yield']}  \n"
                    f"**💰 Market Price:** {info['price']}  \n"
                    f"**🟢 Profit Potential:** {info['profit']}"
                )
            st.markdown("---")

        st.markdown(
            """
              <center><button style="padding:8px 16px; border:none; background:#4CAF50; 
                      color:white; border-radius:8px; cursor:pointer;" 
                      onclick="window.location.reload();">Close</button></center>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -------------------------------
# Show model accuracy
# -------------------------------
st.markdown("---")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.metric("📊 Model Accuracy", f"{acc*100:.2f}%")
