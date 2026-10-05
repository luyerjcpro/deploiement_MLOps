from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
DOSSIER_BRUT = RACINE / "data" / "raw"
DOSSIER_PROCESSED = RACINE / "data" / "processed"


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

    df.to_csv(DOSSIER_PROCESSED / "df_final.csv", index=False)

    return df


if __name__ == "__main__":
    data_sirh, data_eval, data_sondage = load_data()

    df = merge_data(
        data_sirh,
        data_eval,
        data_sondage,
    )
    print("Dataset final créé.")
