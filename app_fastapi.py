
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml_service import load_model, predict_diabetes


model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model()
    yield
    model = None


app = FastAPI(
    title="SmartClinic Diabetes Prediction API",
    description=(
        "Educational machine-learning API for diabetes "
        "risk estimation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class DiabetesFeatures(BaseModel):
    Pregnancies: float = Field(ge=0, le=30)
    Glucose: float = Field(gt=0, le=500)
    BloodPressure: float = Field(gt=0, le=300)
    SkinThickness: float = Field(ge=0, le=150)
    Insulin: float = Field(ge=0, le=3000)
    BMI: float = Field(gt=0, le=100)
    DiabetesPedigreeFunction: float = Field(ge=0, le=10)
    Age: float = Field(ge=1, le=120)


class PredictionRequest(BaseModel):
    features: DiabetesFeatures


@app.get("/")
def home():
    return {
        "message": "Welcome to SmartClinic Diabetes API",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict/diabetes")
def predict(request: PredictionRequest):
    try:
        result = predict_diabetes(
            model,
            request.features.model_dump(),
        )
        return result

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check server logs.",
        )