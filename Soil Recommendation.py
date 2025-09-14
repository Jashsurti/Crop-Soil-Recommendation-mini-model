import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# -------------------------------
# Step 1: Create Dataset
# -------------------------------
data = {
    'Nitrogen': [90, 30, 40, 100, 20, 35],
    'Phosphorus': [40, 60, 50, 40, 70, 45],
    'Potassium': [40, 40, 60, 50, 40, 55],
    'pH': [6.5, 5.5, 6.0, 6.8, 5.2, 6.1],
    'Rainfall': [200, 100, 180, 220, 90, 170],
    'Temperature': [28, 25, 27, 29, 24, 26],
    'SoilType': ['Clay', 'Loamy', 'Sandy', 'Clay', 'Loamy', 'Sandy'],
    'Crop': ['Rice', 'Banana', 'Pepper', 'Rice', 'Banana', 'Pepper']
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -------------------------------
# Step 4: Train Model
# -------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

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

if st.button("🔍 Recommend Crop"):
    sample = [[nitrogen, phosphorus, potassium, ph, rainfall, temperature, soil_enc.transform([soil_type])[0]]]
    pred = model.predict(sample)
    crop = crop_enc.inverse_transform(pred)

    st.success(f"✅ Recommended Crop: **{crop[0]}**")

    # Advisory notes
    if crop[0] == "Rice":
        st.info("🌾 Rice grows best in clay soil with high rainfall. Ensure proper water management.")
    elif crop[0] == "Banana":
        st.info("🍌 Banana thrives in loamy soil with moderate rainfall. Use organic compost for better yield.")
    elif crop[0] == "Pepper":
        st.info("🌶️ Pepper prefers sandy soil with warm climate. Provide support poles and shade trees.")

# Show model accuracy for judges
st.markdown("---")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.metric("📊 Model Accuracy", f"{acc*100:.2f}%")
