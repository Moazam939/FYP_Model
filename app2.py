import json
import os
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "car_price_model.pkl")
VALID_VALUES_PATH = os.path.join(BASE_DIR, "valid_values.json")

print(f"[STARTUP] Looking for model at   : {MODEL_PATH}")
print(f"[STARTUP] Looking for values at  : {VALID_VALUES_PATH}")
print(f"[STARTUP] Model file exists?     : {os.path.exists(MODEL_PATH)}")
print(f"[STARTUP] Values file exists?    : {os.path.exists(VALID_VALUES_PATH)}")

try:
    pipeline = joblib.load(MODEL_PATH)
    print("[STARTUP] Model loaded successfully")
except FileNotFoundError:
    raise RuntimeError(
        f"'{MODEL_PATH}' nahi mili. Pehle save_model.py chalayein taake model save ho."
    )

try:
    with open(VALID_VALUES_PATH, "r", encoding="utf-8") as f:
        VALID_VALUES = json.load(f)
    print(f"[STARTUP] Valid values loaded - keys: {list(VALID_VALUES.keys())}")
except FileNotFoundError:
    print(f"[STARTUP] WARNING: {VALID_VALUES_PATH} not found. Dropdowns will be empty!")
    VALID_VALUES = {}
except json.JSONDecodeError as e:
    print(f"[STARTUP] WARNING: valid_values.json exists but is not valid JSON: {e}")
    VALID_VALUES = {}


CURRENT_YEAR = 2026

# 2. APP INITIALIZATION
app = FastAPI(
    title="AutoIntel Car Price Prediction API",
    description="LightGBM model jo used car ki details se uski market price predict karta hai (PKR mein).",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. INPUT SCHEMA
class CarInput(BaseModel):
    Brand: str = Field(..., example="TOYOTA", description="Car brand (e.g. TOYOTA, SUZUKI, HONDA)")
    Model: str = Field(..., example="COROLLA", description="Car model name")
    Variant: str = Field("BASE", example="GLI", description="Variant/trim (e.g. GLI, VXL, BASE agar pata na ho)")
    Year: int = Field(..., ge=1980, le=2025, example=2020, description="Manufacturing year")
    Millage: float = Field(..., ge=0, le=1000000, example=45000, description="Total driven kilometers")
    Fuel: str = Field(..., example="PETROL", description="Fuel type: PETROL, DIESEL, LPG, HYBRID, ELECTRIC")
    Transmission: str = Field(..., example="AUTOMATIC", description="AUTOMATIC or MANUAL")
    Province: str = Field(..., example="PUNJAB", description="Province/city where car is registered")
    Color: str = Field(..., example="WHITE", description="Car color")
    Assembly: str = Field(..., example="LOCAL", description="LOCAL or IMPORTED")
    Engine_Capacity: float = Field(..., ge=600, le=7000, alias="Engine Capacity",
                                    example=1300, description="Engine capacity in cc")
    Body_Type: str = Field(..., alias="Body Type", example="SEDAN",
                            description="Body type: SEDAN, HATCHBACK, SUV, etc.")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Brand": "TOYOTA",
                "Model": "COROLLA",
                "Variant": "GLI",
                "Year": 2020,
                "Millage": 45000,
                "Fuel": "PETROL",
                "Transmission": "AUTOMATIC",
                "Province": "PUNJAB",
                "Color": "WHITE",
                "Assembly": "LOCAL",
                "Engine Capacity": 1300,
                "Body Type": "SEDAN"
            }
        }

# 4. OUTPUT SCHEMA
class PredictionOutput(BaseModel):
    predicted_price_pkr: float
    predicted_price_formatted: str
    car_age: int
    input_summary: dict

# 5. HELPER: raw input ko model ke expected format mein convert karna
def prepare_features(car: CarInput) -> pd.DataFrame:
    car_age = CURRENT_YEAR - car.Year
    mileage_per_year = car.Millage / car_age if car_age > 0 else car.Millage

    row = {
        "Brand": car.Brand.upper().strip(),
        "Model": car.Model.upper().strip(),
        "Variant": car.Variant.upper().strip(),
        "Year": car.Year,
        "Millage": car.Millage,
        "Fuel": car.Fuel.upper().strip(),
        "Transmission": car.Transmission.upper().strip(),
        "Province": car.Province.upper().strip(),
        "Color": car.Color.upper().strip(),
        "Assembly": car.Assembly.upper().strip(),
        "Engine Capacity": car.Engine_Capacity,
        "Body Type": car.Body_Type.upper().strip(),
        "Car_Age": car_age,
        "Millage_Per_Year": mileage_per_year,
    }
    return pd.DataFrame([row])

# 6. ROUTES
@app.get("/")
def home():
    return {
        "message": "Car Price Prediction API is running.",
        "docs": "/docs",
        "predict_endpoint": "POST /predict"
    }


@app.get("/valid-values")
def get_valid_values():
    """Frontend dropdowns banane ke liye valid categorical options."""
    return VALID_VALUES


@app.post("/predict", response_model=PredictionOutput)
def predict_price(car: CarInput):
    try:
        X_new = prepare_features(car)

        # Model ne log-transformed price par train kiya tha, isliye expm1 se wapis convert
        log_pred = pipeline.predict(X_new)[0]
        price_pred = float(np.expm1(log_pred))
        price_pred = max(price_pred, 0)  # negative price na aaye

        return PredictionOutput(
            predicted_price_pkr=round(price_pred, 2),
            predicted_price_formatted=f"PKR {price_pred:,.0f}",
            car_age=CURRENT_YEAR - car.Year,
            input_summary=car.model_dump(by_alias=True)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": pipeline is not None}