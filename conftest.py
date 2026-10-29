"""Fixtures partagées par tous les fichiers de test.

Point de vigilance central : ce fichier force l'application à travailler sur
une base de TEST, jamais sur la base applicative. Comme `database/connexion_db.py`
lit `DATABASE_URL` et construit son `engine` AU MOMENT DE L'IMPORT, la variable
d'environnement doit être réécrite AVANT le premier import de l'application —
d'où l'ordre inhabituel de ce fichier (code, puis imports marqués `noqa: E402`).

Choix de la base de test, par ordre de priorité :
  1. la variable `TEST_DATABASE_URL` si elle est définie ;
  2. sinon `DATABASE_URL` si son nom de base se termine déjà par `_test`
     (c'est le cas en CI : `projet5_test`) ;
  3. sinon `DATABASE_URL` avec `_test` ajouté au nom de la base.

Création de la base de test en local :

    psql -h localhost -p 5432 -U postgres -c "CREATE DATABASE attrition_test;"
"""

import os
from urllib.parse import urlsplit, urlunsplit

import pytest
from dotenv import load_dotenv

load_dotenv()

# Clé API utilisée par les tests. setdefault : si la variable existe déjà
# (CI, .env), on la respecte ; sinon on en fournit une, faute de quoi
# `api.main` lèverait une ValueError dès l'import.
os.environ.setdefault("SECRET_API_KEY", "cle-de-test-pytest")


def _construire_url_de_test() -> str:
    """Renvoie l'URL de la base de test, ou interrompt la session si elle est douteuse."""
    url_applicative = os.getenv("DATABASE_URL")
    url_explicite = os.getenv("TEST_DATABASE_URL")

    if url_explicite:
        url_retenue = url_explicite
    elif not url_applicative:
        pytest.exit(
            "Ni TEST_DATABASE_URL ni DATABASE_URL ne sont définies : "
            "impossible de savoir sur quelle base lancer les tests.",
            returncode=4,
        )
    else:
        parties = urlsplit(url_applicative)
        nom_base = parties.path.rstrip("/")
        if nom_base.endswith("_test"):
            url_retenue = url_applicative
        else:
            url_retenue = urlunsplit(parties._replace(path=f"{nom_base}_test"))

    # Garde-fou : les tests créent et détruisent le schéma. S'ils pointaient
    # sur la base applicative, `drop_all` effacerait les données réelles.
    if (
        url_applicative
        and url_retenue == url_applicative
        and not urlsplit(url_retenue).path.rstrip("/").endswith("_test")
    ):
        pytest.exit(
            "La base de test est identique à la base applicative. "
            "Définissez TEST_DATABASE_URL vers une base dédiée.",
            returncode=4,
        )

    return url_retenue


URL_DE_TEST = _construire_url_de_test()

# Réécriture AVANT tout import de l'application : `connexion_db` lira celle-ci.
os.environ["DATABASE_URL"] = URL_DE_TEST

# Imports volontairement placés après la réécriture de DATABASE_URL ci-dessus
# (la règle E402 n'est pas active dans la configuration Ruff de ce projet).
from fastapi.testclient import TestClient
from sqlalchemy import text

from api.main import app
from database.connexion_db import SessionLocal, engine, get_session
from database.models_tables import Base, Employee


# ---------------------------------------------------------------------------
# Étage « session » : le schéma n'est construit qu'une fois pour toute la suite
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def schema_de_test():
    """Crée le schéma au début de la session de test, le détruit à la fin.

    Scope 'session' : créer et détruire les tables est coûteux et n'a aucune
    raison d'être refait entre deux tests. Ce qui doit repartir de zéro à
    chaque test, ce sont les LIGNES — voir la fixture `session`.

    `drop_all` en entrée aussi : si une exécution précédente s'est interrompue
    brutalement, la suivante repart quand même d'une base propre.
    """
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    yield

    Base.metadata.drop_all(bind=engine)


# ---------------------------------------------------------------------------
# Étage « function » : chaque test part d'une base vide
# ---------------------------------------------------------------------------
@pytest.fixture
def session():
    """Session SQLAlchemy pour un test, suivie d'un nettoyage des lignes.

    Scope 'function' (le défaut) : c'est ce qui garantit l'indépendance des
    tests. Sans ce nettoyage, un test qui compte les lignes de `predictions`
    dépendrait de ce qui a tourné avant lui, donc de l'ordre d'exécution.

    RESTART IDENTITY remet les séquences à zéro pour que les identifiants
    soient reproductibles ; CASCADE est nécessaire à cause de la clé étrangère
    predictions.employee_id -> employees.id.
    """
    with SessionLocal() as s:
        yield s

    with engine.begin() as connexion:
        connexion.execute(
            text("TRUNCATE predictions, employees RESTART IDENTITY CASCADE")
        )


@pytest.fixture
def client(session):
    """Client HTTP de test, branché sur la session du test.

    `dependency_overrides` remplace la dépendance `get_session` de
    l'application par une fonction qui rend LA session du test : l'API et le
    test partagent donc la même connexion, ce qui permet au test d'intégration
    de relire ce que l'endpoint vient d'écrire.

    `with TestClient(app)` est ici OBLIGATOIRE, contrairement au cas général :
    c'est le gestionnaire de contexte qui déclenche le `lifespan` de main.py,
    lequel charge `modele.joblib` dans le dictionnaire `ressources`. Sans lui,
    `/employees/{id}/predict` lèverait un KeyError sur `ressources["modele_ml"]`.
    """

    def get_session_de_test():
        yield session

    app.dependency_overrides[get_session] = get_session_de_test

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


@pytest.fixture
def entetes_valides():
    """En-tête d'authentification attendu par `verify_api_key`."""
    return {"X-API-Key": os.environ["SECRET_API_KEY"]}


@pytest.fixture
def employe(session):
    """Un employé complet inséré en base, renvoyé avec son `id` auto-généré.

    Toutes les colonnes sont renseignées, y compris les cinq features dérivées
    (`stagnation_promotion` et suivantes) : en production elles sont calculées
    par `create_features` lors du chargement du CSV, et l'endpoint de
    prédiction les relit telles quelles depuis la base.
    """
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
        # Features dérivées, cohérentes avec create_features() :
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


@pytest.fixture
def employee_id_inexistant():
    """Identifiant garanti absent de la base de test."""
    return 999999