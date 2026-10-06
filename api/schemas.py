from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EmployeeInput(BaseModel):
    # Identifiant de l'employé existant en base.
    employee_id: int = Field(gt=0)

    # Variables directement utilisées par le modèle.
    age: int = Field(ge=18, le=60)
    genre: Literal["F", "M"]
    revenu_mensuel: int = Field(ge=1009, le=19999)
    statut_marital: Literal[
        "Célibataire",
        "Divorcé(e)",
        "Marié(e)",
    ]
    poste: Literal[
        "Assistant de Direction",
        "Cadre Commercial",
        "Consultant",
        "Directeur Technique",
        "Manager",
        "Représentant Commercial",
        "Ressources Humaines",
        "Senior Manager",
        "Tech Lead",
    ]
    nombre_experiences_precedentes: int = Field(ge=0, le=9)
    annees_dans_le_poste_actuel: int = Field(ge=0, le=18)
    nombre_participation_pee: int = Field(ge=0, le=3)
    nb_formations_suivies: int = Field(ge=0, le=6)
    niveau_education: Literal[1, 2, 3, 4, 5]
    domaine_etude: Literal[
        "Autre",
        "Entrepreunariat",
        "Infra & Cloud",
        "Marketing",
        "Ressources Humaines",
        "Transformation Digitale",
    ]
    satisfaction_employee_environnement: Literal[1, 2, 3, 4]
    satisfaction_employee_nature_travail: Literal[1, 2, 3, 4]
    satisfaction_employee_equipe: Literal[1, 2, 3, 4]
    satisfaction_employee_equilibre_pro_perso: Literal[1, 2, 3, 4]
    note_evaluation_actuelle: Literal[3, 4]
    augementation_salaire_precedente: Literal[
        "11 %",
        "12 %",
        "13 %",
        "14 %",
        "15 %",
        "16 %",
        "17 %",
        "18 %",
        "19 %",
        "20 %",
        "21 %",
        "22 %",
        "23 %",
        "24 %",
        "25 %",
    ]

    # Variables sources nécessaires à create_features().
    annees_depuis_la_derniere_promotion: int = Field(ge=0, le=15)
    annees_dans_l_entreprise: int = Field(ge=0, le=40)
    distance_domicile_travail: int = Field(ge=1, le=29)
    heure_supplementaires: Literal["Non", "Oui"]
    note_evaluation_precedente: Literal[1, 2, 3, 4]
    frequence_deplacement: Literal[
        "Aucun",
        "Occasionnel",
        "Frequent",
    ]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    employee_id: int = Field(gt=0)
    classe_predite: Literal["Non", "Oui"]
    probabilite_non: float = Field(ge=0.0, le=1.0)
    probabilite_oui: float = Field(ge=0.0, le=1.0)