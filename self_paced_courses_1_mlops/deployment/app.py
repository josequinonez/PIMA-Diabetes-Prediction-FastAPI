from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from huggingface_hub import hf_hub_download
import pandas as pd

# Define the FastAPI application
app = FastAPI()

# Download and load the model
# Replace "josequinonez/PIMA-Diabetes-Prediction-FastAPI" with your actual Hugging Face repo ID if different
latest_name = "best_pima_diabetes_model_latest.joblib"
model_path = hf_hub_download(repo_id="josequinonez/PIMA-Diabetes-Prediction-FastAPI",
                             filename=latest_name, repo_type="model")
model = joblib.load(model_path)

# Define the input data structure using Pydantic
class DiabetesFeatures(BaseModel):
    preg: int
    plas: int
    pres: int
    skin: int
    test: int
    mass: float
    pedi: float
    age: int

# Define the prediction endpoint
@app.post("/predict/")
async def predict_diabetes(features: DiabetesFeatures):
    # Convert input features to a pandas DataFrame
    input_data = pd.DataFrame([features.model_dump()])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Return the prediction result
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    return {"prediction": result}

# Optional: Add a root endpoint for testing
@app.get("/")
async def read_root():
    return {"message": "PIMA Diabetes Prediction API is running!"}
