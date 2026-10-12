"""1) Tests UNITAIRES — une fonction, seule, sans base ni API.

Critère de reconnaissance : aucun test de ce fichier ne demande les fixtures
`client`, `session` ou `employe`. Il n'y a ni PostgreSQL, ni HTTP, ni modèle
sérialisé. Si l'un de ces tests échoue, le défaut est forcément dans
`create_features`, nulle part ailleurs — c'est tout l'intérêt de l'isolation.

`create_features` est le bon candidat au test unitaire : c'est la fonction qui
porte les règles métier du projet, et un écart entre le calcul fait à
l'entraînement et celui fait au chargement des données produirait un modèle
silencieusement faux. Un test rapide vaut mieux qu'une relecture.
"""

import pandas as pd
import pytest

from data.feature_engineering import create_features


def test_create_features_calcule_les_cinq_variables_derivees():
    # Given : un employé aux valeurs simples, choisies pour que chaque calcul
    # soit vérifiable de tête
    df = pd.DataFrame(
        [
            {
                "annees_dans_l_entreprise": 10,
                "annees_depuis_la_derniere_promotion": 2,
                "distance_domicile_travail": 15,
                "heure_supplementaires": "Oui",
                "note_evaluation_actuelle": 4,
                "note_evaluation_precedente": 3,
                "frequence_deplacement": "Frequent",
            }
        ]
    )

    # When : on applique le feature engineering
    resultat = create_features(df)

    # Then : les cinq variables dérivées valent ce qu'on attend
    ligne = resultat.iloc[0]
    assert ligne["stagnation_promotion"] == pytest.approx(0.2)  # 2 / 10
    assert ligne["charge_trajet_travail"] == 15  # 15 × heures sup. (Oui)
    assert ligne["evolution"] == 1  # 4 − 3
    assert ligne["heures_sup_bin"] == 1
    assert ligne["frequence_deplacement_ord"] == 2  # Frequent

    # Et la fonction ne modifie pas le DataFrame reçu (elle fait un .copy())
    assert "stagnation_promotion" not in df.columns


def test_create_features_gere_les_cas_limites():
    # Given : trois employés représentant les cas qui cassent le calcul naïf :
    #  - ancienneté nulle  -> division par zéro
    #  - promotion plus ancienne que l'ancienneté -> ratio supérieur à 1
    #  - pas d'heures supplémentaires -> charge de trajet neutralisée
    df = pd.DataFrame(
        [
            {
                "annees_dans_l_entreprise": 0,
                "annees_depuis_la_derniere_promotion": 0,
                "distance_domicile_travail": 10,
                "heure_supplementaires": "Non",
                "note_evaluation_actuelle": 3,
                "note_evaluation_precedente": 3,
                "frequence_deplacement": "Aucun",
            },
            {
                "annees_dans_l_entreprise": 3,
                "annees_depuis_la_derniere_promotion": 5,
                "distance_domicile_travail": 20,
                "heure_supplementaires": "Oui",
                "note_evaluation_actuelle": 3,
                "note_evaluation_precedente": 4,
                "frequence_deplacement": "Occasionnel",
            },
            {
                "annees_dans_l_entreprise": 8,
                "annees_depuis_la_derniere_promotion": 4,
                "distance_domicile_travail": 25,
                "heure_supplementaires": "Non",
                "note_evaluation_actuelle": 4,
                "note_evaluation_precedente": 4,
                "frequence_deplacement": "Aucun",
            },
        ]
    )

    # When : on applique le feature engineering
    resultat = create_features(df)

    # Then : aucune valeur n'est manquante ni hors bornes
    assert resultat["stagnation_promotion"].notna().all()
    assert resultat["stagnation_promotion"].between(0.0, 1.0).all()

    # Then : cas par cas :
    # ancienneté nulle -> le fillna(0) évite le NaN
    assert resultat.loc[0, "stagnation_promotion"] == pytest.approx(0.0)
    # sans heures supplémentaires, la charge de trajet est annulée
    assert resultat.loc[0, "charge_trajet_travail"] == 0
    assert resultat.loc[0, "heures_sup_bin"] == 0
    assert resultat.loc[0, "frequence_deplacement_ord"] == 0

    # ratio 5/3 > 1 -> le clip(0, 1) le ramène à 1
    assert resultat.loc[1, "stagnation_promotion"] == pytest.approx(1.0)
    # une note en baisse donne une évolution négative
    assert resultat.loc[1, "evolution"] == -1

    assert resultat.loc[2, "charge_trajet_travail"] == 0
    assert resultat.loc[2, "evolution"] == 0