import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# -------------------------------
# Step 1: Expanded Dataset
# -------------------------------
data = {
    'Nitrogen': [
        90, 30, 40, 100, 20, 35, 85, 25, 50, 95,
        60, 70, 55, 45, 65, 75, 80, 30, 20, 40,
        110, 120, 95, 105, 115, 125, 130, 70, 85, 100,
        45, 55, 65, 75, 85, 95, 105, 115, 125, 135
    ],
    'Phosphorus': [
        40, 60, 50, 40, 70, 45, 45, 65, 55, 35,
        50, 55, 60, 65, 70, 75, 45, 50, 55, 60,
        40, 45, 50, 55, 60, 65, 70, 75, 55, 50,
        65, 70, 75, 80, 85, 40, 45, 50, 55, 60
    ],
    'Potassium': [
        40, 40, 60, 50, 40, 55, 45, 50, 50, 40,
        55, 60, 65, 70, 45, 50, 55, 60, 65, 70,
        40, 45, 50, 55, 60, 65, 70, 45, 50, 55,
        60, 65, 70, 40, 45, 50, 55, 60, 65, 70
    ],
    'pH': [
        6.5, 5.5, 6.0, 6.8, 5.2, 6.1, 6.4, 5.6, 6.0, 6.7,
        6.3, 6.2, 6.5, 6.7, 5.8, 5.9, 6.0, 5.5, 5.7, 6.1,
        6.8, 6.9, 7.0, 7.2, 6.5, 6.6, 6.7, 6.0, 6.3, 6.8,
        5.5, 5.8, 6.1, 6.3, 6.7, 6.9, 7.1, 7.2, 6.8, 6.6
    ],
    'Rainfall': [
        200, 100, 180, 220, 90, 170, 210, 120, 160, 230,
        140, 150, 180, 200, 210, 170, 160, 130, 140, 150,
        250, 260, 240, 230, 220, 210, 200, 190, 180, 170,
        150, 160, 170, 180, 190, 200, 220, 230, 240, 250
    ],
    'Temperature': [
        28, 25, 27, 29, 24, 26, 27, 26, 28, 30,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 25,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 30, 29, 28, 27, 26, 25, 24, 23
    ],
    'SoilType': [
        'Clay', 'Loamy', 'Sandy', 'Clay', 'Loamy', 'Sandy', 'Clay', 'Loamy', 'Sandy', 'Clay',
        'Loamy', 'Sandy', 'Clay', 'Loamy', 'Sandy', 'Clay', 'Loamy', 'Sandy', 'Clay', 'Loamy',
        'Sandy', 'Clay', 'Loamy', 'Sandy', 'Clay', 'Loamy', 'Sandy', 'Clay', 'Loamy', 'Sandy',
        'Clay', 'Loamy', 'Sandy', 'Clay', 'Loamy', 'Sandy', 'Clay', 'Loamy', 'Sandy', 'Clay'
    ],
    'Crop': [
        'Rice', 'Banana', 'Pepper', 'Rice', 'Banana', 'Pepper', 'Rice', 'Banana', 'Pepper', 'Rice',
        'Wheat', 'Wheat', 'Wheat', 'Wheat', 'Maize', 'Maize', 'Maize', 'Maize', 'Tomato', 'Tomato',
        'Tomato', 'Tomato', 'Cotton', 'Cotton', 'Cotton', 'Cotton', 'Banana', 'Banana', 'Rice', 'Rice',
        'Pepper', 'Pepper', 'Maize', 'Maize', 'Tomato', 'Tomato', 'Cotton', 'Cotton', 'Wheat', 'Wheat'
    ]
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
# -------------------------------
# Crop Details Dictionary (Kerala Specific)
# -------------------------------
# -------------------------------
# Crop Details Dictionary (Kerala + Fertilizer Dosage)
# -------------------------------
crop_details = {
    "Rice": """🌾 **Rice (Kerala)**
- Soil: Wide range; pH 5.0 – 8.0 acceptable
- Climate: Flowering optimum 16–20°C; grain filling 18–32°C (above 35°C reduces yield)
- Rainfall: 150 – 250 cm annually; grown in Virippu, Mundakan, Puncha seasons
- Nutrients: Kerala soils often acidic (pH 4.5–5.5) & low in K → requires good N & K
- Fertilizers: Urea (N), DAP/SSP (P), MOP (K)
- Dosage: ~90:45:45 kg N:P:K per hectare (split into 3 doses: basal, tillering, panicle initiation)
- Notes: Needs standing water, clayey or loamy soils are best""",

    "Banana": """🍌 **Banana (Kerala)**
- Soil: Well-drained, fertile, depth ≥1 m; avoid soils with pH > 8.0
- Climate: Tropical humid; sea level – 1000 m; optimum temperature ~27°C
- Rainfall: Requires assured moisture (rainfall or irrigation); avoid waterlogging
- Nutrients: Very high demand for N & K; responds well to organic manure
- Fertilizers: Urea (N), DAP (P), MOP (K)
- Dosage: ~200–250 g N, 60–70 g P, 300 g K per plant/year (in 3–4 split doses + FYM 10–15 kg/plant)
- Notes: Protect from strong winds; prefers loamy soil with organic matter""",

    "Pepper": """🌶️ **Black Pepper (Kerala)**
- Soil: Light, porous, well-drained, rich in organic matter; avoid waterlogged heavy soils
- Climate: Warm & humid; optimum 20–30°C; altitude sea level – 1200 m
- Rainfall: ~250 cm, well-distributed; dry spells reduce flowering & yield
- Nutrients: Mature vines (~3+ yrs) need high K, moderate N & P
- Fertilizers: Urea (N), SSP (P), MOP (K)
- Dosage: ~50 g N, 50 g P2O5, 150 g K2O per vine/year (in 2 splits: pre-monsoon & post-monsoon) + 10 kg compost
- Notes: Requires support trees (standards), mulching, and organic manure""",

    "Wheat": """🌾 **Wheat (Kerala – limited)**
- Soil: Loamy/clay loam, well-drained
- Climate: Prefers cooler rabi season; 10–25°C
- Rainfall: 75–120 mm
- Nutrients: Moderate NPK requirement
- Fertilizers: Urea (N), DAP (P), MOP (K)
- Dosage: ~100:50:50 kg N:P:K per hectare (50% N basal, 50% at tillering)
- Notes: Not a major Kerala crop, but trial areas exist in drier zones""",

    "Maize": """🌽 **Maize (Kerala)**
- Soil: Fertile sandy loam, well-drained
- Climate: Optimum 18–27°C
- Rainfall: 50–100 mm; requires irrigation in dry spells
- Nutrients: High nitrogen demand; balanced P & K
- Fertilizers: Urea (N), DAP (P), MOP (K)
- Dosage: ~120:60:40 kg N:P:K per hectare (split: basal + top dressing at knee-high & tasseling stage)
- Notes: Used as fodder + grain crop, grown in pockets in Kerala""",

    "Tomato": """🍅 **Tomato (Kerala)**
- Soil: Sandy loam, rich in organic matter
- Climate: Optimum 20–25°C
- Rainfall: 50–125 mm; avoid heavy rains (disease risk)
- Nutrients: Requires high P & K, moderate N
- Fertilizers: Urea (N), DAP/SSP (P), MOP (K) + FYM
- Dosage: ~100:50:50 kg N:P:K per hectare + 20 t FYM (split N & K into 2–3 applications)
- Notes: Best in dry season (Nov–Feb); needs staking & plant protection""",

    "Cotton": """🌱 **Cotton (Kerala – very limited)**
- Soil: Black soils (regur) or sandy loam, well-drained
- Climate: Warm (21–30°C)
- Rainfall: 50–100 mm; does poorly in heavy monsoon areas
- Nutrients: Needs high K, moderate N & P
- Fertilizers: Urea (N), SSP/DAP (P), MOP (K)
- Dosage: ~80:40:40 kg N:P:K per hectare (1/2 N basal, 1/2 top dressing at flowering)
- Notes: Not widely grown in Kerala due to high rainfall & humidity"""
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
