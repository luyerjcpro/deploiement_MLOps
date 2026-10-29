
import pytest
from pydantic import ValidationError

from api.schemas import EmployeeFeatures
from database.models_tables import Prediction

# 1. TEST UNITAIRE : PYDANTIC

def test_unitaire_pydantic(employe):
    donnees = {
        champ: getattr(employe, champ)
        for champ in EmployeeFeatures.model_fields
    }

    # Données correctes : validation réussie.
    profil = EmployeeFeatures(**donnees)
    assert profil.age == 35

    # Données incorrectes : validation refusée.
    donnees["age"] = 17

    with pytest.raises(ValidationError):
        EmployeeFeatures(**donnees)


# 2. TEST FONCTIONNEL : GET ET POST

def test_fonctionnel(client, employe, entetes_valides):
    # GET : consulter un employé.
    reponse_get = client.get(
        f"/employees/{employe.id}",
        headers=entetes_valides,
    )

    assert reponse_get.status_code == 200
    assert reponse_get.json()["id"] == employe.id

    # POST : calculer une prédiction.
    reponse_post = client.post(
        f"/employees/{employe.id}/predict",
        headers=entetes_valides,
    )

    assert reponse_post.status_code == 201

    resultat = reponse_post.json()

    assert resultat["classe_predite"] in ("Oui", "Non")
    assert 0 <= resultat["probabilite_oui"] <= 1
    assert 0 <= resultat["probabilite_non"] <= 1


# 3. TEST D'INTÉGRATION : POSTGRESQL

def test_integration(client, session, employe, entetes_valides):
    # Exécuter une prédiction.
    reponse = client.post(
        f"/employees/{employe.id}/predict",
        headers=entetes_valides,
    )

    assert reponse.status_code == 201

    # Relire la prédiction dans PostgreSQL.
    identifiant = reponse.json()["id"]
    session.expire_all()
    prediction = session.get(Prediction, identifiant)

    assert prediction is not None
    assert prediction.employee_id == employe.id
    assert prediction.classe_predite == (
        reponse.json()["classe_predite"]
    )


# 4. TEST DE BOUT EN BOUT : CONSULTATION

def test_bout_en_bout(client, employe, entetes_valides):
    # Parcours : client -> API -> PostgreSQL -> client.
    reponse = client.get(
        f"/employees/{employe.id}",
        headers=entetes_valides,
    )

    assert reponse.status_code == 200

    donnees = reponse.json()

    assert donnees["id"] == employe.id
    assert donnees["age"] == employe.age
    assert donnees["genre"] == employe.genre
    assert donnees["revenu_mensuel"] == employe.revenu_mensuel


# 5. TEST DE SÉCURITÉ : CLÉ API

def test_securite(client, employe, entetes_valides):
    url = f"/employees/{employe.id}"

    # Clé valide : accès autorisé.
    autorise = client.get(url, headers=entetes_valides)
    assert autorise.status_code == 200

    # Clé absente : accès refusé.
    absent = client.get(url)
    assert absent.status_code == 401

    # Clé invalide : accès refusé.
    invalide = client.get(
        url,
        headers={"X-API-Key": "mauvaise-cle"},
    )
    assert invalide.status_code == 401
