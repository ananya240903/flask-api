from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import csv
import re

app = FastAPI()

# Load dataset from GitHub
df = pd.read_csv("https://raw.githubusercontent.com/ananya240903/flask-api/refs/heads/main/signal%20db3.csv")
df.columns = df.columns.str.strip()

# Encode categorical columns
le_sim = LabelEncoder()
le_location = LabelEncoder()
df["Sim"] = le_sim.fit_transform(df["Sim"])
df["Location"] = le_location.fit_transform(df["Location"])

# Mappings for human-readable names
location_mapping = dict(zip(le_location.transform(le_location.classes_), le_location.classes_))
sim_mapping = dict(zip(le_sim.transform(le_sim.classes_), le_sim.classes_))

# Train KNN model
X = df[["Location", "Distance to Tower (km)", "SNR", "Sim"]]
y = df["Strength"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

# --------------------------- MODELS ---------------------------

class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    signal_strength: int
    preferred_sim: str = ""

class QueryRequest(BaseModel):
    query: str

class PredictionRequest(BaseModel):
    Location: str
    Distance: float
    SNR: float
    Sim: str

# --------------------------- UTILS ---------------------------

def log_query(user_query: str, matched_sim: str = ""):
    with open("chat_queries.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), user_query, matched_sim])

# --------------------------- ROUTES ---------------------------

@app.get("/")
def home():
    return {"message": "FastAPI is running on Render!"}

@app.post("/suggest_location")
def suggest_better_location(request: LocationRequest):
    if request.preferred_sim:
        try:
            sim_id = le_sim.transform([request.preferred_sim.lower()])[0]
            filtered_df = df[df["Sim"] == sim_id]
            if not filtered_df.empty:
                best_index = filtered_df["Strength"].idxmax()
                best_location = location_mapping[filtered_df.loc[best_index, "Location"]]
                return {
                    "better_location": best_location,
                    "best_sim": request.preferred_sim.capitalize()
                }
        except Exception as e:
            return {"better_location": "unknown", "best_sim": "unknown", "error": str(e)}

    best_index = df["Strength"].idxmax()
    best_location = location_mapping[df.loc[best_index, "Location"]]
    best_sim = sim_mapping[df.loc[best_index, "Sim"]]
    return {
        "better_location": best_location,
        "best_sim": best_sim
    }

@app.post("/send_query")
def send_query(request: QueryRequest):
    # Clean and lowercase query, remove punctuation
    query = re.sub(r'[^\w\s]', '', request.query.lower())
    known_sims = [sim.lower() for sim in le_sim.classes_]

    matched_sim = None
    for sim in known_sims:
        if sim in query:
            matched_sim = sim
            break

    log_query(request.query, matched_sim or "none")

    if matched_sim:
        return {"response": f"The best location for {matched_sim.capitalize()} is being fetched now."}
    else:
        return {"response": "Please ask for a SIM provider like Jio, Airtel, Vi, or BSNL."}

@app.post("/predict_signal")
def predict_signal(request: PredictionRequest):
    try:
        loc_id = le_location.transform([request.Location.strip().lower()])[0]
        sim_id = le_sim.transform([request.Sim.strip().lower()])[0]

        input_data = [[loc_id, request.Distance, request.SNR, sim_id]]
        prediction = knn.predict(input_data)[0]

        return {"PredictedStrength": prediction}
    except Exception as e:
        return {"PredictedStrength": -1, "error": str(e)}
