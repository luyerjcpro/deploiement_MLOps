import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Stagnation de promotion
    df["stagnation_promotion"] = (
        df["annees_depuis_la_derniere_promotion"]
        / df["annees_dans_l_entreprise"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)

    # Charge trajet / travail
    df["charge_trajet_travail"] = (
        df["distance_domicile_travail"]
        * (df["heure_supplementaires"] == "Oui").astype(int)
    )

    # Évolution de la note
    df["evolution"] = (
        df["note_evaluation_actuelle"]
        - df["note_evaluation_precedente"]
    )

    # Heures supplémentaires
    df["heures_sup_bin"] = (
        df["heure_supplementaires"]
        .map({"Oui": 1, "Non": 0})
    )

    # Fréquence de déplacement
    mapping = {
        "Aucun": 0,
        "Occasionnel": 1,
        "Frequent": 2,
    }

    df["frequence_deplacement_ord"] = (
        df["frequence_deplacement"]
        .map(mapping)
    )

    return df