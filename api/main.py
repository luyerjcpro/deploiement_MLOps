from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "modele.joblib"

ressources: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ressources["modele_ml"] = joblib.load(MODEL_PATH)
    yield
    ressources.clear()


app = FastAPI(
    title="API de prédiction d'attrition",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "message": "API de prédiction d'attrition opérationnelle"
    }