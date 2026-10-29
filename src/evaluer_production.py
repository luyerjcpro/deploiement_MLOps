"""Compare les prédictions servies par l'API aux valeurs réelles du jeu de validation."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score

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

import joblib

MODEL_PATH = PROJECT_ROOT / "models" / "modele.joblib"  # adapte le chemin

LABELS = ["Non", "Oui"]

def afficher_matrice(y_true: pd.Series, y_pred: pd.Series) -> None:
    """Affiche la matrice de confusion en effectifs, en pourcentages et case par case."""
    # 1. Effectifs avec noms d'axes et totaux
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=pd.Index(LABELS, name="Réel"),
        columns=pd.Index(LABELS, name="Prédit"),
    )
    cm_df["Total"] = cm_df.sum(axis=1)
    cm_df.loc["Total"] = cm_df.sum()
 
    print("Matrice de confusion (effectifs) :")
    print(cm_df.to_string())
 
    # 2. Pourcentages par classe réelle (chaque ligne somme à 100 %)
    cm_pct = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=LABELS, normalize="true"),
        index=pd.Index(LABELS, name="Réel"),
        columns=pd.Index(LABELS, name="Prédit"),
    )
    print("\nEn % de chaque classe réelle :")
    print((cm_pct * 100).round(1).astype(str).add(" %").to_string())
 
    # 3. Les quatre cases en langage métier
    tn, fp, fn, tp = cm.ravel()
    print(f"\nVrais négatifs  (reste, prédit reste)  : {tn}")
    print(f"Faux positifs   (reste, prédit départ) : {fp}")
    print(f"Faux négatifs   (part, non détecté)    : {fn}   <- le coût principal")
    print(f"Vrais positifs  (part, détecté)        : {tp}")


def verifier_coherence(comp: pd.DataFrame) -> None:
    """Compare les probabilités de l'API à celles du modèle chargé localement."""
    artefact = joblib.load(MODEL_PATH)
    pipeline = artefact["pipeline"]
    idx_oui = list(artefact["classes"]).index("Oui")
    print(f"\nSeuil enregistré dans le modèle : {artefact['seuil']:.3f}")

    df_val = pd.read_csv(VALIDATE_PATH)
    X_val = df_val.drop(columns=["a_quitte_l_entreprise"])  # mêmes colonnes qu'en entrée de train.py
    df_val["proba_locale"] = pipeline.predict_proba(X_val)[:, idx_oui]

    controle = df_val[["id_employee", "proba_locale"]].merge(
        comp[["id_employee", "probabilite_oui"]], on="id_employee"
    )
    ecart = (controle["proba_locale"] - controle["probabilite_oui"]).abs()
    print(f"Écart max API vs modèle local : {ecart.max():.6f}")
    print(f"Lignes avec écart > 0.001 : {(ecart > 0.001).sum()}")


def main() -> None:
    pred = pd.read_sql(PRED_SQL, engine)
    verite = pd.read_csv(VALIDATE_PATH)[["id_employee", "a_quitte_l_entreprise"]]

    comp = verite.merge(pred, on="id_employee", validate="one_to_one")
    y_true = comp["a_quitte_l_entreprise"]
    y_pred = comp["classe_predite"]


    print(f"{len(comp)} / {len(verite)} employés scorés\n")

    afficher_matrice(y_true, y_pred)

    print("\nRapport de classification :")
    print(classification_report(
        y_true, y_pred,
        labels=LABELS,
        digits=3,
        zero_division=0,
    ))

    ap = average_precision_score(y_true == "Oui", comp["probabilite_oui"])
    print(f"Average precision (indépendante du seuil) : {ap:.3f}")

    verifier_coherence(comp)



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