from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load and preprocess the dataset
# Get the current directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(BASE_DIR, "signal_db3.csv")

df = pd.read_csv(CSV_FILE_PATH)
  # Ensure this file is uploaded to GitHub
df.columns = df.columns.str.strip()

# Encode categorical data
le_sim = LabelEncoder()
le_location = LabelEncoder()
df["Sim"] = le_sim.fit_transform(df["Sim"])
df["Location"] = le_location.fit_transform(df["Location"])

# Train KNN model
X = df[["Location", "Distance to Tower (km)", "SNR", "Sim"]]
y = df["Strength"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

@app.route('/')
def home():
    return "Flask API is running!"

@app.route('/predict', methods=['POST'])
def predict_location():
    data = request.json
    location = le_location.transform([data["Location"]])[0]
    distance = data["Distance"]
    snr = data["SNR"]
    sim = le_sim.transform([data["Sim"]])[0]

    features = np.array([[location, distance, snr, sim]])
    prediction = knn.predict(features)[0]
    
    return jsonify({"Predicted Strength": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
