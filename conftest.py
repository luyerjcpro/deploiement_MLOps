import os
from urllib.parse import urlsplit

import pytest
from dotenv import load_dotenv

load_dotenv()

# Choisir exclusivement une base de test.
url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")

if not url or not urlsplit(url).path.rstrip("/").endswith("_test"):
    pytest.exit(
        "Configurer TEST_DATABASE_URL vers une base dédiée *_test.",
        returncode=4,
    )

# À définir avant les imports de l'application.
os.environ["DATABASE_URL"] = url
os.environ.setdefault("SECRET_API_KEY", "cle-de-test-pytest")

from fastapi.testclient import TestClient
from sqlalchemy import text

from api.main import app
from database.connexion_db import (
    SessionLocal,
    engine,
    get_session,
)
from database.models_tables import Base, Employee


# 1. Préparer les tables de test.
@pytest.fixture(scope="session")
def schema_de_test():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    yield

    Base.metadata.drop_all(bind=engine)


# 2. Fournir une session PostgreSQL.
@pytest.fixture
def session(schema_de_test):
    with SessionLocal() as db:
        yield db

    # Nettoyage après chaque test.
    with engine.begin() as connexion:
        connexion.execute(
            text(
                "TRUNCATE predictions, employees "
                "RESTART IDENTITY CASCADE"
            )
        )


# 3. Fournir le client HTTP FastAPI.
@pytest.fixture
def client(session):

    app.dependency_overrides[get_session] = lambda: session

    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_session, None)


# 4. Fournir un employé enregistré dans PostgreSQL.
@pytest.fixture
def employe(session):

    employe = Employee(
        id_employee=1001,
        age=35,
        genre="M",
        revenu_mensuel=4500,
        statut_marital="Marié(e)",
        poste="Consultant",
        nombre_experiences_precedentes=3,
        annees_dans_le_poste_actuel=4,
        nombre_participation_pee=2,
        nb_formations_suivies=3,
        niveau_education=3,
        domaine_etude="Infra & Cloud",
        satisfaction_employee_environnement=3,
        satisfaction_employee_nature_travail=2,
        satisfaction_employee_equipe=3,
        satisfaction_employee_equilibre_pro_perso=2,
        note_evaluation_actuelle=3,
        augementation_salaire_precedente="14 %",
        annees_dans_l_entreprise=7,
        annees_depuis_la_derniere_promotion=3,
        distance_domicile_travail=18,
        heure_supplementaires="Oui",
        note_evaluation_precedente=3,
        frequence_deplacement="Occasionnel",
        stagnation_promotion=3 / 7,
        charge_trajet_travail=18,
        evolution=0,
        heures_sup_bin=1,
        frequence_deplacement_ord=1,
    )

    session.add(employe)
    session.commit()
    session.refresh(employe)

    return employe


# 5. Fournir une clé API valide.
@pytest.fixture
def entetes_valides():
    return {
        "X-API-Key": os.environ["SECRET_API_KEY"]
    }