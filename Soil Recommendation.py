import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
# Step 5: Evaluate
# -------------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# -------------------------------
# Step 6: Predict for a Farmer
# -------------------------------
# Example: Soil is loamy, Nitrogen=25, P=65, K=45, pH=5.4, Rain=120mm, Temp=25°C
sample = [[25, 65, 45, 5.4, 120, 25, soil_enc.transform(['Loamy'])[0]]]
pred = model.predict(sample)
crop = crop_enc.inverse_transform(pred)

print("Recommended Crop:", crop[0])
