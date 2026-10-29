from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RACINE = Path(__file__).resolve().parents[1]
DOSSIER_BRUT = RACINE / "data" / "raw"
DOSSIER_PROCESSED = RACINE / "data" / "processed"

RANDOM_STATE = 42

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    data_sirh = pd.read_csv(DOSSIER_BRUT / "extrait_sirh.csv")
    data_eval = pd.read_csv(DOSSIER_BRUT / "extrait_eval.csv")
    data_sondage = pd.read_csv(DOSSIER_BRUT / "extrait_sondage.csv")

    return data_sirh, data_eval, data_sondage

def merge_data(
    data_sirh: pd.DataFrame,
    data_eval: pd.DataFrame,
    data_sondage: pd.DataFrame,
) -> pd.DataFrame:

    # Extraction de l'identifiant
    data_eval["id_jointure"] = (
        data_eval["eval_number"]
        .str.extract(r"E_(\d+)")
        .astype(int)
    )

    # Création d'une clé commune
    data_sirh["id_jointure"] = data_sirh["id_employee"]
    data_sondage["id_jointure"] = data_sondage["code_sondage"]

    # Jointures
    df = (
        data_sirh
        .merge(
            data_sondage,
            on="id_jointure",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            data_eval,
            on="id_jointure",
            how="inner",
            validate="one_to_one",
        )
    )

    df=df.drop(columns=["id_jointure", "code_sondage", "eval_number"])

    return df



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


if __name__ == "__main__":
    data_sirh, data_eval, data_sondage = load_data()

    df = merge_data(
        data_sirh,
        data_eval,
        data_sondage,
    )

    df = create_features(df.copy())

    df_final, df_validate = train_test_split(
        df,
        test_size=0.1,
        random_state=RANDOM_STATE,
        stratify=df["a_quitte_l_entreprise"],
    )


    df_final.to_csv(DOSSIER_PROCESSED / "df_final.csv", index=False)
    df_validate.to_csv(DOSSIER_PROCESSED / "df_validate.csv", index=False)

    print(f"df_final    : {len(df_final)} lignes (entraînement + test)")
    print(f"df_validate : {len(df_validate)} lignes (validation via API)")
