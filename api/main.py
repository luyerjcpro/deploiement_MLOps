from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from schemas import EmployeeInput, PredictionResponse
from sqlalchemy.orm import Session

from database.connexion_db import get_session
from database.models_tables import Employee, Prediction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "modele.joblib"

ressources: dict = {}


# Chargement du modèle
@asynccontextmanager
async def lifespan(app: FastAPI):
    ressources["modele_ml"] = joblib.load(MODEL_PATH)
    yield
    ressources.clear()

# Instanciation de l'api
app = FastAPI(
    title="API de prédiction d'attrition",
    lifespan=lifespan,
)

# Session de communication avec la base de données
SessionDep = Annotated[Session, Depends(get_session)]

# Endpoint de contrôle
@app.get("/")
def root():
    return {
        "message": "API de prédiction d'attrition opérationnelle"
    }

# Lecture des données sur la table Employees
@app.get("/employees/{employee_id}")
def read_employee(
    employee_id: int,
    session: SessionDep,
):
    employee = session.get(Employee, employee_id)

    if employee is None:
        raise HTTPException(
            status_code=404,
            detail="Employé introuvable",
        )

    return employee


# Envoie de la requête de prédiction en entrant des données
@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=201,
)
def predire(
    entree: EmployeeInput,
    session: SessionDep,
):
    # La prédiction doit être liée à un employé existant.
    employee = session.get(Employee, entree.employee_id)

    if employee is None:
        raise HTTPException(
            status_code=404,
            detail="Employé introuvable",
        )

    modele_ml = ressources["modele_ml"]

    pipeline = modele_ml["pipeline"]
    seuil = modele_ml["seuil"]

    # employee_id sert à PostgreSQL, pas au modèle.
    donnees_modele = entree.model_dump(
        exclude={"employee_id"}
    )
    df = pd.DataFrame([donnees_modele])

    probabilites = pipeline.predict_proba(df)[0]

    index_non = list(pipeline.classes_).index(0)
    index_oui = list(pipeline.classes_).index(1)

    probabilite_non = float(
        probabilites[index_non]
    )
    probabilite_oui = float(
        probabilites[index_oui]
    )

    classe_predite = (
        "Oui"
        if probabilite_oui >= seuil
        else "Non"
    )

    prediction = Prediction(
        employee_id=entree.employee_id,
        classe_predite=classe_predite,
        probabilite_non=probabilite_non,
        probabilite_oui=probabilite_oui,
    )

    session.add(prediction)
    session.commit()
    session.refresh(prediction)

    return prediction