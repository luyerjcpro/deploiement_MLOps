import json
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "df_final.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "modele.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "modele_metadata.json"
sys.path.insert(0, str(PROJECT_ROOT))

RANDOM_STATE = 42
THRESHOLD = 0.231  # threshold optimal pour le modèle de classification


from data.feature_engineering import create_features

# Colonnes réellement utilisées par le modèle après create_features().
NOMINAL_COLS = [
    "genre",
    "statut_marital",
    "poste",
    "domaine_etude",
    "augementation_salaire_precedente",
]

NUM_COLS = [
    "age",
    "revenu_mensuel",
    "nombre_experiences_precedentes",
    "annees_dans_le_poste_actuel",
    "nombre_participation_pee",
    "nb_formations_suivies",
    "niveau_education",
    "satisfaction_employee_environnement",
    "satisfaction_employee_nature_travail",
    "satisfaction_employee_equipe",
    "satisfaction_employee_equilibre_pro_perso",
    "note_evaluation_actuelle",
    "stagnation_promotion",
    "charge_trajet_travail",
    "evolution",
    "frequence_deplacement_ord",
]

# Séparation des données entre variables explicatives et variable à expliquer
def prepare_data(df: pd.DataFrame):
    le_target = LabelEncoder()

    y = le_target.fit_transform(df["a_quitte_l_entreprise"])
    X = df.drop(columns=["a_quitte_l_entreprise"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, le_target


# Préparation et standardisation des données 
def build_preprocessor() -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "nominal",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="first",
                ),
                NOMINAL_COLS,
            ),
            (
                "num",
                StandardScaler(),
                NUM_COLS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    preprocessor.set_output(transform="pandas")
    return preprocessor


# Pipeline de Imbalanced Learn pour la classification de classes déséquilibrées
# et contenant la fonction create_features
def build_pipeline(preprocessor: ColumnTransformer) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            (
                "features",
                FunctionTransformer(
                    create_features,
                    validate=False,
                ),
            ),
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "clf",
                GradientBoostingClassifier(
                    random_state=RANDOM_STATE,
                    learning_rate=0.1,
                    max_depth=3,
                    min_samples_leaf=5,
                    n_estimators=100,
                ),
            ),
        ]
    )


# Entrainement du modèle
def train_model(df: pd.DataFrame):
    X_train, X_test, y_train, y_test, le_target = prepare_data(df)

    preprocessor = build_preprocessor()
    pipeline = build_pipeline(preprocessor)

    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)

    return pipeline, le_target, accuracy


# Sérialisation du modèle entrainé
def save_model(
    model,
    le_target,
    threshold: float,
    path: Path = MODEL_PATH,
) -> None:
    modele_ml = {
        "pipeline": model,
        "seuil": threshold,
        "classes": le_target.classes_,
    }

    joblib.dump(modele_ml, path)


# Enregistrement des métadonnées du modèle sérialisé
def save_metadata(
    pipeline,
    le_target,
    threshold: float,
    accuracy: float,
    df: pd.DataFrame,
    path: Path = METADATA_PATH,
) -> None:
    classifier = pipeline.named_steps["clf"]
    smote = pipeline.named_steps["smote"]

    metadata = {
        "model": {
            "name": classifier.__class__.__name__,
            "parameters": classifier.get_params(),
            "random_state": RANDOM_STATE,
        },
        "training": {
            "trained_at_utc": datetime.now(UTC).isoformat(),
            "n_rows": len(df),
            "n_raw_features": len(df.columns) - 1,
            "test_size": 0.2,
            "accuracy_test": accuracy,
        },
        "classification": {
            "target": "a_quitte_l_entreprise",
            "classes": le_target.classes_.tolist(),
            "threshold": threshold,
        },
        "features": {
            "nominal": NOMINAL_COLS,
            "numerical": NUM_COLS,
        },
        "preprocessing": {
            "categorical_encoder": "OneHotEncoder",
            "numerical_scaler": "StandardScaler",
            "sampling": smote.__class__.__name__,
            "sampling_parameters": smote.get_params(),
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
            "imbalanced_learn": version("imbalanced-learn"),
            "joblib": version("joblib"),
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    
    pipeline, le_target, accuracy = train_model(df)

    print(f"Accuracy test : {accuracy:.3f}")
    print(f"Classes cible : {le_target.classes_}")
    print(f"Seuil : {THRESHOLD}")

    # Vérification : une ligne brute doit être prédictible directement.
    X_brut = df.drop(columns=["a_quitte_l_entreprise"])
    exemple = X_brut.iloc[[0]]

    prediction = pipeline.predict(exemple)
    probabilites = pipeline.predict_proba(exemple)

    print(f"Prédiction test : {prediction}")
    print(f"Probabilités test : {probabilites}")

    # Sérialisation du modèle
    save_model(
        pipeline,
        le_target,
        THRESHOLD,
    )

    # Métadonnées du modèle format JSON
    save_metadata(
        pipeline,
        le_target,
        THRESHOLD,
        accuracy,
        df,
    )

    print(f"Modèle sauvegardé : {MODEL_PATH}")
    print(f"Métadonnées sauvegardées : {METADATA_PATH}")
