import pytest
from fastapi.testclient import TestClient

from api.main import app
from database.connexion_db import engine
from database.models_tables import Base


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    Base.metadata.create_all(bind=engine)

    yield

    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


@pytest.fixture
def employee_id():
    return 1


@pytest.fixture
def payload_employee(employee_id):
    return {
        "employee_id": employee_id,
        "age": 35,
        "genre": "M",
        "revenu_mensuel": 4500,
        "statut_marital": "Marié(e)",
        "poste": "Consultant",
        "nombre_experiences_precedentes": 3,
        "annees_dans_le_poste_actuel": 4,
        "nombre_participation_pee": 2,
        "nb_formations_suivies": 3,
        "niveau_education": 3,
        "domaine_etude": "Infra & Cloud",
        "satisfaction_employee_environnement": 3,
        "satisfaction_employee_nature_travail": 2,
        "satisfaction_employee_equipe": 3,
        "satisfaction_employee_equilibre_pro_perso": 2,
        "note_evaluation_actuelle": 3,
        "augementation_salaire_precedente": "14 %",
        "annees_depuis_la_derniere_promotion": 3,
        "annees_dans_l_entreprise": 7,
        "distance_domicile_travail": 18,
        "heure_supplementaires": "Oui",
        "note_evaluation_precedente": 3,
        "frequence_deplacement": "Occasionnel",
    }


@pytest.fixture
def employee_id_inexistant():
    return 999999