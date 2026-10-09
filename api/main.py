import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "modele.joblib"

from api.schemas import EmployeeInput, PredictionResponse
from database.connexion_db import get_session
from database.models_tables import Employee, Prediction

#######  X-API-KEY  #######
# Charge le fichier .env si l'api est en local
load_dotenv()

# Récupère la clé secrète depuis l'environnement du serveur
SECRET_API_KEY = os.getenv("SECRET_API_KEY")

# Sécurité pour bloquer l'API si le secret est absent au démarrage
if not SECRET_API_KEY:
    raise ValueError(
        "La variable d'environnement SECRET_API_KEY est manquante."
    )


#######  MODELE  #######
# Dictionnaire ressources pour enregistrer le modèle
ressources: dict = {}

# Chargement du modèle
@asynccontextmanager
async def lifespan(app: FastAPI):
    ressources["modele_ml"] = joblib.load(MODEL_PATH)
    yield
    ressources.clear()

#######  DOCUMENTATION  #######
# Ajout de la description de l'api dans son swagger.
description = """
🚀 **Employee Predictor API** vous permet d'évaluer les indicateurs des employés grâce au Machine Learning.

## Fonctionnalités principales
* **Consultation des données employés**
* **Prédiction en temps réel** (`/predict`)
"""

# Tags des routes de l'api.
tags_metadata = [
    {
        "name": "Monitoring",
             "description": "Indicateurs techniques pour surveiller la santé de l'API et vérifier sa disponibilité.",
    },
    {
        "name": "Employees",
        "description": "Gestion et consultation des données relatives aux collaborateurs.",
    },
    {
        "name": "Machine Learning",
        "description": "Calculs de scores, prédictions et analyses statistiques via nos modèles.",
    },
]

# Modèle pour documenter la structure de l'erreur dans Swagger
class MessageErreur(BaseModel):
    detail: str


#######  API  #######
# Instanciation de l'api
app = FastAPI(
    title="API de prédiction d'attrition",
    openapi_tags=tags_metadata, # Associe les métadonnées
    description=description,
    lifespan=lifespan,
)

# Session de communication avec la base de données
SessionDep = Annotated[Session, Depends(get_session)]


#######  ROUTES  #######
# Endpoint de contrôle
@app.get("/", tags=["Monitoring"])
def root():
    """Endpoint racine - confirme que l'API est en ligne."""
    return {
        "message": "API de prédiction d'attrition opérationnelle"
    }

@app.get("/health", tags=["Monitoring"])
def health_check():
    """Healthcheck explicite, utilisé par les plateformes de déploiement."""
    return {"status": "healthy"}

# Endpint pour la lecture des données sur la table Employees
@app.get(
    "/employees/{employee_id}", 
    tags=["Employees"],
    responses={
        404: {
            "model": MessageErreur, 
            "description": "L'employé avec cet identifiant n'existe pas en base de données."
        }
    }
)
def read_employee(
    employee_id: int,
    session: SessionDep,
):
    """ Affiche les informations de l'employé à partir de son identifiant.  
    """

    employee = session.get(Employee, employee_id)

    if employee is None:
        raise HTTPException(
            status_code=404,
            detail="Employé introuvable",
        )

    return employee


# Endpoint pour l'envoi de la requête de prédiction en entrant des données
@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=201,
    tags=["Machine Learning"],
)
def predire(
    entree: EmployeeInput,
    session: SessionDep,
):
    """Génère et enregistre une prédiction de Machine Learning pour un employé.

    Cette fonction extrait les données de l'employé, applique le pipeline de
    classification, calcule les probabilités selon un seuil personnalisé, puis
    persiste le résultat dans la base de données.

    ### Args:
    - **entree (EmployeeInput)**: 
        Les caractéristiques nécessaires au modèle, contenant obligatoirement l'identifiant de l'employé.  
    - **session (SessionDep)**: 
        La session de base de données injectée par dépendance.

    ### Raises:
    -   HTTPException: Erreur 404 si l'identifiant de l'employé fourni n'existe pas dans la base de données.

    ### Returns:
    -   **Prediction**: L'objet prédiction enregistré en base contenant la classe finale ainsi que le détail des probabilités.
    """

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