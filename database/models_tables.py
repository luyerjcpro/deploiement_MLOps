from sqlalchemy import CheckConstraint, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Employee(Base):
    __tablename__ = "employees"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    age: Mapped[int]
    genre: Mapped[str]
    revenu_mensuel: Mapped[int]
    statut_marital: Mapped[str]
    poste: Mapped[str]
    nombre_experiences_precedentes: Mapped[int]
    annees_dans_le_poste_actuel: Mapped[int]
    nombre_participation_pee: Mapped[int]
    nb_formations_suivies: Mapped[int]
    niveau_education: Mapped[int]
    domaine_etude: Mapped[str]
    satisfaction_employee_environnement: Mapped[int]
    satisfaction_employee_nature_travail: Mapped[int]
    satisfaction_employee_equipe: Mapped[int]
    satisfaction_employee_equilibre_pro_perso: Mapped[int]
    note_evaluation_actuelle: Mapped[int]
    augementation_salaire_precedente: Mapped[str]


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    employee_id: Mapped[int] = mapped_column(
        ForeignKey("employees.id"),
        nullable=False,
    )

    classe_predite: Mapped[str]
    probabilite_non: Mapped[float]
    probabilite_oui: Mapped[float]

    __table_args__ = (
        CheckConstraint(
            "classe_predite IN ('Oui', 'Non')",
            name="check_prediction_class",
        ),
        CheckConstraint(
            "probabilite_non >= 0 AND probabilite_non <= 1",
            name="check_probability_non",
        ),
        CheckConstraint(
            "probabilite_oui >= 0 AND probabilite_oui <= 1",
            name="check_probability_oui",
        ),
    )