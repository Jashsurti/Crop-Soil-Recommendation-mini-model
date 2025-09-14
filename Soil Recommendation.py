import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# -------------------------------
# Step 1: Expanded Dataset
# -------------------------------
data = {
    'Nitrogen': [90, 30, 40, 100, 20, 35, 85, 25, 50, 95],
    'Phosphorus': [40, 60, 50, 40, 70, 45, 45, 65, 55, 35],
    'Potassium': [40, 40, 60, 50, 40, 55, 45, 50, 50, 40],
    'pH': [6.5, 5.5, 6.0, 6.8, 5.2, 6.1, 6.4, 5.6, 6.0, 6.7],
    'Rainfall': [200, 100, 180, 220, 90, 170, 210, 120, 160, 230],
    'Temperature': [28, 25, 27, 29, 24, 26, 27, 26, 28, 30],
    'SoilType': ['Clay', 'Loamy', 'Sandy', 'Clay', 'Loamy', 'Sandy', 'Clay', 'Loamy', 'Sandy', 'Clay'],
    'Crop': ['Rice', 'Banana', 'Pepper', 'Rice', 'Banana', 'Pepper', 'Rice', 'Banana', 'Pepper', 'Rice']
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
# Step 3: Train RandomForest
# -------------------------------
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X, y)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Crop Recommendation", page_icon="🌱", layout="centered")
st.title("🌱 Smart Crop Recommendation System")
st.write("Enter soil and weather details to get personalized crop suggestions.")

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
# Crop Details Dictionary
# -------------------------------
crop_details = {
    "Rice": "🌾 Rice grows best in clay soil with high rainfall. Prefers pH 5.5–6.8. Needs standing water and warm temperatures.",
    "Banana": "🍌 Banana thrives in loamy soil with moderate rainfall and warm temperature.",
    "Pepper": "🌶️ Pepper prefers sandy or loamy soil with good drainage, partial shade, and warm climate."
}

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🔍 Recommend Crop"):
    sample = [[nitrogen, phosphorus, potassium, ph, rainfall, temperature, soil_enc.transform([soil_type])[0]]]
    probs = model.predict_proba(sample)[0]

    # Show all crops sorted by probability
    top_idx = probs.argsort()[::-1]

    st.success("✅ Crop Recommendations (based on probability):")
    for i in top_idx:
        crop_name = crop_enc.inverse_transform([i])[0]
        st.write(f"**{crop_name}** - Confidence: {probs[i]*100:.2f}%")
        st.write(crop_details.get(crop_name, "No details available."))
        st.write("---")

# -------------------------------
# Show model accuracy on training data
# -------------------------------
st.markdown("### Model Accuracy")
st.write("⚠️ Note: Small dataset → Accuracy not reliable for production.")
y_pred = model.predict(X)
acc = (y_pred == y).mean()
st.metric("📊 Training Data Accuracy", f"{acc*100:.2f}%")
