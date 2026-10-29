"""2) fonctionnels, 3) intégration, 4) paramétré — tous passent par l'API.

Aucune fixture n'est définie ici : `client`, `session`, `employe`,
`entetes_valides` et `employee_id_inexistant` viennent du `conftest.py` à la
racine du dépôt et sont réclamées par leur seul nom en paramètre.
"""

import pytest
from sqlalchemy import select

from database.models_tables import Prediction


# ===========================================================================
# 2) Tests FONCTIONNELS — le point de vue de l'utilisateur de l'API.
#    Les assertions ne portent QUE sur la réponse HTTP : code de statut et
#    contenu du JSON. Ce que fait l'application en interne ne les regarde pas.
# ===========================================================================
def test_racine_confirme_que_l_api_est_en_ligne(client):
    # Given : une API démarrée
    # When : on appelle l'endpoint public de contrôle
    reponse = client.get("/")

    # Then : elle répond sans exiger d'authentification
    assert reponse.status_code == 200
    assert reponse.json() == {
        "message": "API de prédiction d'attrition opérationnelle"
    }


def test_lecture_d_un_employe_existant(client, entetes_valides, employe):
    # Given : un employé présent en base
    # When : on demande sa fiche avec une clé API valide
    reponse = client.get(
        f"/employees/{employe.id}",
        headers=entetes_valides,
    )

    # Then : l'API renvoie bien cet employé
    assert reponse.status_code == 200
    assert reponse.json()["id"] == employe.id
    assert reponse.json()["id_employee"] == employe.id_employee


def test_prediction_renvoie_un_resultat_exploitable(
    client, entetes_valides, employe
):
    # Given : un employé présent en base
    # When : on demande une prédiction pour cet employé
    reponse = client.post(
        f"/employees/{employe.id}/predict",
        headers=entetes_valides,
    )

    # Then : la ressource est créée et le corps respecte le contrat annoncé
    # dans PredictionResponse
    assert reponse.status_code == 201

    corps = reponse.json()
    assert corps["employee_id"] == employe.id
    assert corps["classe_predite"] in {"Oui", "Non"}
    assert 0.0 <= corps["probabilite_non"] <= 1.0
    assert 0.0 <= corps["probabilite_oui"] <= 1.0

    # Les deux probabilités sont complémentaires : c'est la propriété qui
    # trahirait une inversion d'indice entre les classes 0 et 1.
    assert corps["probabilite_non"] + corps["probabilite_oui"] == pytest.approx(1.0)


def test_prediction_sur_un_employe_inexistant_renvoie_404(
    client, entetes_valides, employee_id_inexistant
):
    # Given : un identifiant absent de la base
    # When : on demande une prédiction pour cet identifiant
    reponse = client.post(
        f"/employees/{employee_id_inexistant}/predict",
        headers=entetes_valides,
    )

    # Then : l'API refuse proprement, sans remonter d'exception
    assert reponse.status_code == 404
    assert reponse.json() == {"detail": "Employé introuvable"}


# ===========================================================================
# 3) Test d'INTÉGRATION — la couture API <-> PostgreSQL.
#    Même point d'entrée que le test fonctionnel ci-dessus, mais l'assertion
#    porte sur l'AUTRE CÔTÉ de la couture : ce qui a réellement été écrit en
#    base. On interroge la table avec SQLAlchemy, pas avec un second appel
#    HTTP — sinon on testerait encore le contrat de l'API.
# ===========================================================================
def test_la_prediction_est_persistee_en_base(
    client, session, entetes_valides, employe
):
    # Given : une table `predictions` vide (garanti par le nettoyage de la
    # fixture `session`)
    assert session.scalars(select(Prediction)).all() == []

    # When : on appelle l'endpoint de prédiction
    reponse = client.post(
        f"/employees/{employe.id}/predict",
        headers=entetes_valides,
    )
    assert reponse.status_code == 201

    # Then : exactement une ligne a été écrite...
    lignes = session.scalars(select(Prediction)).all()
    assert len(lignes) == 1

    ligne = lignes[0]

    # ...rattachée au bon employé par la clé étrangère...
    assert ligne.employee_id == employe.id

    # ...et dont le contenu correspond exactement à la réponse rendue au
    # client : c'est la garantie qu'on ne journalise pas autre chose que ce
    # qu'on annonce.
    corps = reponse.json()
    assert ligne.id == corps["id"]
    assert ligne.classe_predite == corps["classe_predite"]
    assert ligne.probabilite_non == pytest.approx(corps["probabilite_non"])
    assert ligne.probabilite_oui == pytest.approx(corps["probabilite_oui"])


def test_deux_appels_produisent_deux_lignes(
    client, session, entetes_valides, employe
):
    # Given : un employé en base
    # When : on demande deux fois une prédiction pour lui
    for _ in range(2):
        reponse = client.post(
            f"/employees/{employe.id}/predict",
            headers=entetes_valides,
        )
        assert reponse.status_code == 201

    # Then : la table est un journal : elle conserve les deux appels, et le
    # modèle étant déterministe, les deux résultats sont identiques
    lignes = session.scalars(select(Prediction)).all()
    assert len(lignes) == 2
    assert lignes[0].classe_predite == lignes[1].classe_predite
    assert lignes[0].probabilite_oui == pytest.approx(lignes[1].probabilite_oui)


# ===========================================================================
# 4) Test PARAMÉTRÉ — une seule fonction, six tests indépendants.
#    Deux décorateurs `parametrize` empilés produisent le PRODUIT des deux
#    listes : 3 scénarios d'authentification × 2 routes protégées.
#    À noter : la paramétrisation est une technique d'écriture, pas un type de
#    test — celui-ci reste un test fonctionnel. Il couvre le critère
#    « gestion des secrets et scénarios d'erreur » du cahier des charges.
# ===========================================================================
@pytest.mark.parametrize(
    "entetes",
    [
        pytest.param({}, id="sans-entete"),
        pytest.param({"X-API-Key": ""}, id="cle-vide"),
        pytest.param({"X-API-Key": "mauvaise-cle"}, id="cle-invalide"),
    ],
)
@pytest.mark.parametrize(
    "methode, gabarit_url",
    [
        pytest.param("GET", "/employees/{id}", id="lecture-employe"),
        pytest.param("POST", "/employees/{id}/predict", id="prediction"),
    ],
)
def test_les_routes_protegees_refusent_une_cle_invalide(
    client, employe, methode, gabarit_url, entetes
):
    # Given : une route protégée et un en-tête d'authentification défaillant
    url = gabarit_url.format(id=employe.id)

    # When : on appelle la route sans clé valide
    reponse = client.request(methode, url, headers=entetes)

    # Then : l'accès est refusé AVANT toute lecture en base ou appel au modèle
    assert reponse.status_code == 401
    assert "X-API-Key" in reponse.json()["detail"]