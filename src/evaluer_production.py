"""Compare les prédictions servies par l'API aux valeurs réelles du jeu de validation."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

VALIDATE_PATH = PROJECT_ROOT / "data" / "processed" / "df_validate.csv"

from database.connexion_db import engine

PRED_SQL = """
    SELECT DISTINCT ON (p.employee_id)
           e.id_employee, p.classe_predite, p.probabilite_oui
    FROM predictions p
    JOIN employees e ON e.id = p.employee_id
    ORDER BY p.employee_id, p.id DESC
"""


def main() -> None:
    pred = pd.read_sql(PRED_SQL, engine)
    verite = pd.read_csv(VALIDATE_PATH)[["id_employee", "a_quitte_l_entreprise"]]

    comp = verite.merge(pred, on="id_employee", validate="one_to_one")

    print(f"{len(comp)} / {len(verite)} employés scorés\n")
    print(confusion_matrix(
        comp["a_quitte_l_entreprise"], comp["classe_predite"],
        labels=["Non", "Oui"],
    ))
    print()
    print(classification_report(
        comp["a_quitte_l_entreprise"], comp["classe_predite"],
        labels=["Non", "Oui"],
    ))


if __name__ == "__main__":
    main()


# Dans un premier terminal, lancer l'api avec uvicorn et entrer la clé de sécurité dans authorize
# Dans un nouveau terminal, copier le code si dessous pour faire apparaître la matrice de confusion

# export API_KEY=$(grep '^SECRET_API_KEY=' .env | cut -d= -f2- | tr -d '"'"'")

# for i in $(seq 1 147); do
#   curl -sf -X POST "http://127.0.0.1:port_a_entrer/employees/$i/predict" \
#        -H "X-API-Key: $API_KEY" > /dev/null || echo "échec sur $i"
# done

# unset API_KEY