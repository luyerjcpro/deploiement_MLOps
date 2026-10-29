from pathlib import Path

import pandas as pd
from connexion_db import SessionLocal
from models_tables import Employee
from sqlalchemy import func, select

BASE_DIR = Path(__file__).resolve().parents[1]

CSV_PATH = BASE_DIR / "data" / "processed" / "df_validate.csv"




def lire_csv(chemin: str) -> pd.DataFrame:
    df = pd.read_csv(chemin)

    colonnes = [
        colonne.name
        for colonne in Employee.__table__.columns
        if colonne.name != "id"
    ]

    colonnes_manquantes = set(colonnes) - set(df.columns)

    if colonnes_manquantes:
        raise ValueError(
            f"Colonnes absentes du CSV : {sorted(colonnes_manquantes)}"
        )

    return (
        df[colonnes]
        .astype(object)
        .where(pd.notna(df[colonnes]), None)
    )


def charger() -> int:
    df = lire_csv(CSV_PATH)

    with SessionLocal() as session:
        if session.scalar(select(func.count()).select_from(Employee)):
            print("La table employees contient déjà des données. Chargement ignoré.")
            return 0

        session.add_all(Employee(**ligne) for ligne in df.to_dict(orient="records"))
        session.commit()

    return len(df)


if __name__ == "__main__":
    nombre_insertions = charger()

    print(
        f"{nombre_insertions} employés insérés dans PostgreSQL."
    )