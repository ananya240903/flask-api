from fastapi import FastAPI
import pandas as pd
import numpy as np
from pydantic import BaseModel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Load dataset from GitHub
df = pd.read_csv("https://raw.githubusercontent.com/ananya240903/flask-api/refs/heads/main/signal%20db3.csv")  # Replace with your GitHub link

df.columns = df.columns.str.strip()
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

class PredictionRequest(BaseModel):
    Location: str
    Distance: float
    SNR: float
    Sim: str

@app.get("/")
def home():
    return {"message": "FastAPI is running on Render!"}

@app.post("/predict")
def predict(request: PredictionRequest):
    location = le_location.transform([request.Location])[0]
    distance = request.Distance
    snr = request.SNR
    sim = le_sim.transform([request.Sim])[0]

    features = np.array([[location, distance, snr, sim]])
    prediction = knn.predict(features)[0]

    return {"Predicted Strength": prediction}
