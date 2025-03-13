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

location_mapping = dict(zip(le_location.transform(le_location.classes_), le_location.classes_))
sim_mapping = dict(zip(le_sim.transform(le_sim.classes_), le_sim.classes_))

# Train KNN model
X = df[["Location", "Distance to Tower (km)", "SNR", "Sim"]]
y = df["Strength"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    signal_strength: int

@app.get("/")
def home():
    return {"message": "FastAPI is running on Render!"}

@app.post("/suggest_location")
def suggest_better_location(request: LocationRequest):
    # Find the best nearby location
    best_location_index = df["Strength"].idxmax()
    best_location_name = location_mapping[df.loc[best_location_index, "Location"]]
    best_sim_id = df.loc[best_location_index, "Sim"]
    best_sim_name = sim_mapping[best_sim_id]

    return {
        "better_location": best_location_name,
        "best_sim": best_sim_name
    }
