# Déploiement d'un modèle de machine learning : prédiction de l'attrition

> **objectif et contexte** :  
Futurisys, client fictif de ce projet, souhaite passer d'un modèle de machine learning développé dans un notebook à un service que d'autres applications peuvent interroger. 

>Le cas d'usage retenu est la prédiction de l'attrition : estimer, à partir des données RH, la probabilité qu'un collaborateur quitte l'entreprise. Les équipes RH peuvent ainsi anticiper les départs et engager un échange avec les personnes concernées avant qu'il ne soit trop tard.

>Le modèle, un Gradient Boosting, a été entraîné sur trois extractions RH : le SIRH, les évaluations annuelles et un sondage interne. Son seuil de décision (0,231) est volontairement bas : il privilégie la détection des départs, quitte à émettre quelques fausses alertes.

L'API rend ce modèle utilisable :

- elle renvoie la fiche d'un employé enregistré en base ;
- elle calcule, pour cet employé, la probabilité de départ et la classe prédite (Oui ou Non) ;
- elle enregistre chaque prédiction dans PostgreSQL, ce qui permet de tracer les résultats et d'évaluer le modèle sur des employés qu'il n'a pas vus pendant l'entraînement ;
- elle n'est accessible qu'avec une clé API.

Ce dépôt est une preuve de concept. Il démontre la chaîne complète de mise en production (préparation des données, entraînement, base de données, API, tests, intégration continue et déploiement) sur un jeu de 147 employés de validation. Ce n'est pas un outil RH prêt à l'emploi : une prédiction est une aide à la décision, à interpréter par les RH, et non une décision automatique.


## Sommaire

1. [Arborescence du dépôt](#1-arborescence-du-dépôt)
2. [Prérequis](#2-prérequis)
3. [Installation et configuration](#3-installation-et-configuration)
4. [Lancer le projet](#4-lancer-le-projet)
5. [Utiliser l'API](#5-utiliser-lapi)
6. [Tests](#6-tests)
7. [Intégration et déploiement continus](#7-intégration-et-déploiement-continus)
8. [Évaluation du modèle](#8-évaluation-du-modèle)
9. [Architecture](#9-architecture)

## 1. Arborescence du dépôt

```text
deploiement_MLOps/
├── api/
│   ├── main.py                 # application FastAPI, routes, chargement du modèle
│   ├── schemas.py              # schémas Pydantic (entrée du modèle, réponse)
│   └── security.py             # vérification de la clé API (en-tête X-API-Key)
├── data/
│   ├── feature_engineering.py  # create_features : variables métier
│   ├── load_clean_data.py
│   ├── load_clean_fe_data.py   # fusion, nettoyage, variables métier, découpage 90/10
│   ├── raw/                    # extractions RH brutes (à créer, non versionné)
│   └── processed/              # df_final.csv, df_validate.csv (à créer, non versionné)
├── database/
│   ├── connexion_db.py         # engine et sessions SQLAlchemy
│   ├── models_tables.py        # tables employees et predictions
│   ├── create_tables_db.py     # création des tables
│   └── insert_raws.py          # chargement de df_validate.csv dans employees
├── models/
│   ├── pipeline.py             # entraînement du pipeline
│   ├── modele.joblib           # pipeline entraîné et seuil de décision
│   └── modele_metadata.json    # hyperparamètres, variables, métriques
├── src/
│   └── evaluer_production.py   # matrice de confusion et métriques sur les prédictions servies
├── tests/
│   └── test_cinq_types.py
├── conftest.py                 # fixtures pytest
├── .github/workflows/ci.yaml   # intégration continue
├── Dockerfile
├── docker-compose.yaml
├── .env.example                # modèle du fichier .env (à copier)
├── pyproject.toml
├── uv.lock
└── requirements.txt
```

**Organisation du projet** : Le dépôt est organisé par couche, dans l'ordre de la chaîne de traitement : 
- `data/` prépare les données, 
- `models/` entraîne le modèle et conserve l'artefact, 
- `database/` décrit et remplit la base, 
- `api/` expose le modèle.  

Chaque couche s'appuie sur ce que produisent les précédentes : l'API, par exemple, ne réentraîne jamais le modèle, elle charge le fichier `modele.joblib` produit par `models/pipeline.py`. On peut ainsi modifier ou tester une couche sans toucher aux autres.

## 2. Prérequis

- Git
- [uv](https://docs.astral.sh/uv/), qui installe au besoin Python 3.13, la version exigée par le projet
- PostgreSQL 16, pour le lancement local sans Docker
- Docker et Docker Compose, pour le lancement conteneurisé

## 3. Installation et configuration

### 3.1 Cloner le dépôt et installer les dépendances

```bash
git clone https://github.com/luyerjcpro/deploiement_MLOps.git
cd deploiement_MLOps
uv sync
```

Sans uv, les mêmes versions s'installent avec `pip install -r requirements.txt`.

### 3.2 Configurer les secrets

Les secrets sont lus dans un fichier `.env` à la racine. Ce fichier n'est pas versionné.

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"   # génère une clé API
```

Compléter `.env` (sans guillemets ni espace autour du `=`) :

```dotenv
# Base applicative, lancement local
DATABASE_URL=postgresql+psycopg://postgres:motdepasse@localhost:5432/attrition

# Base de test : obligatoire pour pytest, son nom doit se terminer par _test
TEST_DATABASE_URL=postgresql+psycopg://postgres:motdepasse@localhost:5432/attrition_test

# Clé exigée dans l'en-tête X-API-Key (caractères ASCII uniquement)
SECRET_API_KEY=coller_ici_la_cle_generee

# Lancement avec Docker Compose
POSTGRES_USER=postgres
POSTGRES_PASSWORD=motdepasse
POSTGRES_DB=attrition
# POSTGRES_PORT=5433   # si le port 5432 est déjà occupé par un PostgreSQL local
```
#### Rôle de chaque variable

| Variable | Lue par | Rôle |
| --- | --- | --- |
| `DATABASE_URL` | `database/connexion_db.py` | Adresse de la base applicative. Elle sert à créer le moteur SQLAlchemy dès l'import du module : sans elle, ni l'API ni les scripts de base de données ne démarrent. |
| `TEST_DATABASE_URL` | `conftest.py` | Adresse de la base de test. pytest la substitue à `DATABASE_URL` avant d'importer l'application. Le nom de la base doit se terminer par `_test`, car les tests y suppriment et recréent les tables. |
| `SECRET_API_KEY` | `api/main.py`, `api/security.py` | Clé d'accès aux routes protégées. L'API refuse de démarrer sans elle, puis la compare, à chaque requête, à l'en-tête `X-API-Key`. Elle ne doit contenir que des caractères ASCII. |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | Docker Compose | Identifiants avec lesquels le conteneur PostgreSQL est initialisé. Compose s'en sert aussi pour construire l'adresse de la base transmise au conteneur de l'API. Le code Python ne les lit pas. |
| `POSTGRES_PORT` (facultative) | Docker Compose | Port de la machine hôte sur lequel le conteneur PostgreSQL est publié (5432 par défaut). |

Le fichier `.env` est chargé par `load_dotenv()`. Une variable déjà définie dans l'environnement reste prioritaire sur `.env` : en intégration continue et sur Render, il n'y a pas de fichier `.env`, et ce sont ces plateformes qui fournissent les mêmes variables.

Le fichier `.env` n'est pas versionné (il figure dans `.gitignore`), car il contient des mots de passe et la clé API. Le fichier `.env.example`, lui, est versionné : il donne le modèle à copier, sans aucune valeur réelle.

#### Lecture de `DATABASE_URL`

`postgresql+psycopg://postgres:motdepasse@localhost:5432/attrition`

| Partie | Signification |
| --- | --- |
| `postgresql` | le SGBD, c'est-à-dire le dialecte SQL à produire |
| `+psycopg` | le pilote, c'est-à-dire la bibliothèque Python qui transmet les requêtes |
| `postgres:motdepasse` | l'utilisateur et son mot de passe |
| `localhost:5432` | l'hôte et le port du serveur PostgreSQL |
| `attrition` | le nom de la base |

SQLAlchemy se place entre le code Python et la base, et il a besoin de deux informations distinctes. Le SGBD lui indique quel dialecte SQL produire : PostgreSQL, MySQL et SQLite n'ont ni les mêmes types de données ni tout à fait la même syntaxe. Le pilote est la bibliothèque qui ouvre la connexion réseau et transmet les requêtes au serveur, selon le protocole propre à PostgreSQL. Comme un même SGBD peut être joint par plusieurs pilotes (psycopg2, psycopg 3, asyncpg…), l'adresse prend la forme `dialecte+pilote`.


### 3.3 Préparer les données

```bash
mkdir -p data/raw data/processed
```

Placer les trois extractions RH dans `data/raw/`, sous ces noms exacts : `extrait_sirh.csv`, `extrait_eval.csv` et `extrait_sondage.csv`.

**Liens de téléchargement des trois fichiers** : 

- [Extrait SIRH](https://s3.eu-west-1.amazonaws.com/course.oc-static.com/projects/1047_Data+Scientist+ML/P4_DSML_1047/extrait_sirh.csv)  
- [Extrait évaluations de performance](https://s3.eu-west-1.amazonaws.com/course.oc-static.com/projects/1047_Data+Scientist+ML/P4_DSML_1047/extrait_eval.csv)  
- [Extrait sondage](https://s3.eu-west-1.amazonaws.com/course.oc-static.com/projects/1047_Data+Scientist+ML/P4_DSML_1047/extrait_sondage.csv)  

```bash
uv run python data/load_clean_fe_data.py
```

Le script produit deux fichiers dans `data/processed/` :

- `df_final.csv` (90 % des employés), qui sert à entraîner le modèle ;
- `df_validate.csv` (10 %, soit 147 employés), qui est chargé en base et sert à évaluer les prédictions de l'API.



### 3.4 Réentraîner le modèle (facultatif)

`models/modele.joblib` est déjà versionné : cette étape ne sert qu'à le régénérer.

```bash
uv run python models/pipeline.py
```

Le script produit `models/modele.joblib` et `models/modele_metadata.json`.

## 4. Lancer le projet

### 4.1 En local, sans Docker

Créer les bases (le mot de passe doit correspondre à celui de `.env`) :

```bash
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'motdepasse';"
sudo -u postgres psql -c "CREATE DATABASE attrition;"
sudo -u postgres psql -c "CREATE DATABASE attrition_test;"
```

Créer les tables, puis charger les employés :

```bash
uv run python database/create_tables_db.py
uv run python database/insert_raws.py   # 147 employés ; relancé, le script ne crée pas de doublons
```

Démarrer l'API :

```bash
uv run uvicorn api.main:app --reload --port 8000
```


### 4.2 En local, avec Docker Compose

```bash
docker compose up --build -d
docker compose run --rm api python database/create_tables_db.py
uv run python database/insert_raws.py
```

La dernière commande s'exécute depuis la machine hôte, car les CSV ne sont pas copiés dans l'image. Elle utilise le `DATABASE_URL` de `.env`, qui doit pointer vers `localhost` et vers le port publié par le conteneur `db` (5432, ou la valeur de `POSTGRES_PORT`).


```bash
docker compose down      # arrête les conteneurs, les données sont conservées
docker compose down -v   # arrête les conteneurs et supprime le volume de la base
```

### 4.3 En ligne, sur Render

API : <https://deploiement-mlops.onrender.com/docs>


## 5. Utiliser l'API

| Méthode | Route | Clé API | Réponses |
| --- | --- | --- | --- |
| GET | `/` | non | 200 : message de bienvenue |
| GET | `/health` | non | 200 : `{"status": "healthy"}` |
| GET | `/employees/{employee_id}` | oui | 200 : fiche de l'employé · 401, 404, 422 |
| POST | `/employees/{employee_id}/predict` | oui | 201 : prédiction enregistrée · 401, 404, 422 |

**Avec Swagger** : cliquer sur **Authorize**, coller la valeur de `SECRET_API_KEY`, puis tester les routes. Les identifiants valides vont de 1 à 147. Il s'agit de la colonne `id` de la table `employees`, et non de `id_employee`.

**Avec curl** :

```bash
API_KEY=$(grep '^SECRET_API_KEY=' .env | cut -d= -f2-)

curl -H "X-API-Key: $API_KEY" http://localhost:8000/employees/1
curl -X POST -H "X-API-Key: $API_KEY" http://localhost:8000/employees/1/predict
```

Forme de la réponse de prédiction (valeurs d'exemple) :

```json
{"id": 1, "employee_id": 1, "classe_predite": "Oui", "probabilite_non": 0.65, "probabilite_oui": 0.35}
```

**Effet du seuil** : la classe vaut « Oui » dès que `probabilite_oui` atteint la valeur du seuil 0,231.

**Vérifier l'enregistrement en base** :

```bash
# PostgreSQL local
psql -h localhost -U postgres -d attrition -c "SELECT * FROM predictions ORDER BY id DESC LIMIT 5;"

# Docker Compose
docker compose exec db psql -U postgres -d attrition -c "SELECT * FROM predictions ORDER BY id DESC LIMIT 5;"
```

## 6. Tests

```bash
uv run pytest -v
```

Les tests utilisent la base désignée par `TEST_DATABASE_URL`. Si son nom ne se termine pas par `_test`, pytest s'arrête avant de toucher à quoi que ce soit.

| Type | Ce qui est protégé |
| --- | --- |
| Unitaire | les schémas Pydantic |
| Fonctionnel | les routes GET et POST |
| Intégration | l'enregistrement des prédictions dans PostgreSQL |
| Bout en bout | le parcours complet d'un utilisateur, de l'API à la base |
| Sécurité | l'accès par clé API |

Fixtures (dans `conftest.py`) : `schema_de_test`, `session`, `client`, `employe`, `entetes_valides`.

### 6.1 Démarche

Les tests ne cherchent pas à tout couvrir : ils protègent les comportements dont la défaillance rendrait le service faux ou dangereux. Leur rôle est de détecter une régression, c'est-à-dire une modification future qui casserait un comportement attendu. Les bibliothèques utilisées (FastAPI, Pydantic, SQLAlchemy, PostgreSQL) sont déjà testées par leurs auteurs : ce qui est testé ici, c'est la façon dont le projet s'en sert.

Chaque type de test du tableau ci-dessus cible un risque précis. La suite repose sur quatre principes :

- **Une vraie base PostgreSQL plutôt qu'une base simulée** : les contraintes et les types de données se comportent comme en production.
- **Des tests indépendants** : chaque test part de tables vides, et son résultat ne dépend pas de l'ordre d'exécution.
- **Des données réelles protégées** : la suite refuse de démarrer si le nom de la base visée ne se termine pas par `_test`, car elle supprime et recrée les tables.
- **Des limites assumées** : la suite vérifie les cas principaux (validation des données, réponses 200, 201 et 401, enregistrement en base). D'autres scénarios d'erreur, comme les réponses 404 et 422, peuvent s'ajouter sur le même modèle.

### 6.2 Fixtures

Les fixtures, définies dans `conftest.py`, préparent puis nettoient ce dont les tests ont besoin. pytest fournit une fixture à un test d'après le nom de ses paramètres, et exécute le nettoyage (le code placé après `yield`) une fois le test terminé.

| Fixture | Portée | Dépend de | Rôle |
| --- | --- | --- | --- |
| `schema_de_test` | toute la suite | aucune | Supprime puis recrée les tables au début de la suite, et les supprime à la fin. Une exécution interrompue ne gêne donc pas la suivante. |
| `session` | chaque test | `schema_de_test` | Ouvre une session SQLAlchemy sur la base de test. Après le test, vide les tables `predictions` et `employees` et remet les identifiants à 1 (`TRUNCATE … RESTART IDENTITY`). |
| `client` | chaque test | `session` | Fournit un client HTTP (`TestClient`). Remplace la dépendance `get_session` de l'API par la session du test : l'API travaille sur la base de test, et le test peut relire ce qu'elle a écrit. Le bloc `with` déclenche le `lifespan` de l'API, donc le chargement du modèle. |
| `employe` | chaque test | `session` | Insère un employé complet (30 colonnes, dont les 5 variables dérivées, cohérentes avec `create_features`) et le renvoie avec son identifiant généré. |
| `entetes_valides` | chaque test | aucune | Renvoie l'en-tête `X-API-Key` avec la clé attendue par l'API : celle du fichier `.env`, ou une clé de test par défaut. |

Avant même les fixtures, `conftest.py` choisit la base de test et s'arrête si son nom ne se termine pas par `_test`. Il réécrit ensuite `DATABASE_URL` avant d'importer l'application, parce que le moteur SQLAlchemy est créé dès l'import de `database/connexion_db.py`.

### 6.3 Couverture des tests HTML

Commandes pour générer les couvertures des tests sous format HTML.

Linux :
```bash
uv run pytest --cov=api --cov=database --cov-report=html
xdg-open htmlcov/index.html
```
Windows (PowerShell):
```bash
uv run pytest --cov=api --cov=database --cov-report=html
start htmlcov\index.html
```

## 7. Intégration et déploiement continus

Le workflow `.github/workflows/ci.yaml` se déclenche à chaque `push` et enchaîne trois jobs :

| Job | Rôle |
| --- | --- |
| `lint` | vérification du code avec Ruff |
| `test` | exécution de pytest contre un service PostgreSQL 16 éphémère (base `projet5_test`) |
| `build` | construction de l'image Docker, seulement si `lint` et `test` réussissent |

La clé API de la CI provient des secrets du dépôt GitHub (`SECRET_API_KEY`).


## 8. Évaluation du modèle

L'API doit être démarrée (section 4.1 ou 4.2). On demande d'abord une prédiction pour chacun des 147 employés de validation, puis on compare ces prédictions aux départs réels :

```bash
API_KEY=$(grep '^SECRET_API_KEY=' .env | cut -d= -f2-)

for i in $(seq 1 147); do
  curl -sf -X POST "http://localhost:8000/employees/$i/predict" \
       -H "X-API-Key: $API_KEY" > /dev/null || echo "échec sur l'employé $i"
done

uv run python src/evaluer_production.py
```

Le script affiche :

- la matrice de confusion ;
- le rapport de classification ;
- l'average precision ;
- un contrôle de cohérence entre les probabilités servies par l'API et celles du modèle chargé localement.

**Résultats** :
```bash
Matrice de confusion (effectifs) :
Prédit  Non  Oui  Total
Réel                   
Non     100   23    123
Oui      11   13     24
Total   111   36    147

En % de chaque classe réelle :
Prédit     Non     Oui
Réel                  
Non     81.3 %  18.7 %
Oui     45.8 %  54.2 %

Vrais négatifs  (reste, prédit reste)  : 100
Faux positifs   (reste, prédit départ) : 23
Faux négatifs   (part, non détecté)    : 11   <- le coût principal
Vrais positifs  (part, détecté)        : 13

Rapport de classification :
              precision    recall  f1-score   support

         Non      0.901     0.813     0.855       123
         Oui      0.361     0.542     0.433        24

    accuracy                          0.769       147
   macro avg      0.631     0.677     0.644       147
weighted avg      0.813     0.769     0.786       147

Average precision (indépendante du seuil) : 0.385

Seuil enregistré dans le modèle : 0.231
Écart max API vs modèle local : 0.000000
Lignes avec écart > 0.001 : 0
```
-> Sur les 24 départs réels, l’API en signale 13, soit un rappel de 54 %.  
-> Sur les 36 alertes émises, 13 sont de vrais départs, soit une précision de 36 %.  

Au hasard, on tomberait juste 16 fois sur 100 (24/147).  
**Le modèle fait donc environ 2,2 fois mieux que le hasard.**

## 9. Architecture client-serveur

Le projet suit une architecture client-serveur à trois niveaux : le client, l'API et la base de données. En local, les trois tournent sur la même machine et communiquent par l'adresse `localhost`.

### 9.1 Vision globale
Le client (Swagger UI, curl ou un navigateur) envoie des requêtes HTTP à l'API FastAPI, servie par Uvicorn sur le port 8000. L'API est le seul composant qui accède à la base : elle vérifie la clé `X-API-Key`, valide les données avec Pydantic, puis lit ou écrit dans PostgreSQL au moyen de SQLAlchemy et du pilote psycopg. PostgreSQL est un serveur distinct, qui écoute sur son port par défaut, le 5432. Le modèle de machine learning, lui, n'est pas un service séparé : il est chargé une seule fois au démarrage, dans la mémoire du processus de l'API, et n'ouvre donc aucun port.

| Liaison | Protocole | Adresse | Défini dans |
| --- | --- | --- | --- |
| Client → API | HTTP | `http://localhost:8000` | la commande `uvicorn ... --port 8000` |
| API → PostgreSQL | protocole PostgreSQL (TCP) | `localhost:5432` | la variable `DATABASE_URL` du fichier `.env` |

En local, les échanges ne quittent pas la machine : le HTTP non chiffré suffit.

### 9.2 Base de données 
Chiffres clés :

- 28 variables en entrée du modèle : 23 issues des extractions RH, plus 5 créées par `create_features` ;
- 30 colonnes dans la table `employees` : ces 28 variables, plus `id` et `id_employee` ;
- en sortie : une classe (Oui ou Non) et deux probabilités, enregistrées dans la table `predictions` ;
- un seuil de décision de 0,231. 


```mermaid
%%{init: {
  "fontFamily":  "Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif",
  "fontSize": 18
}}%%

erDiagram
    employees ||--o{ predictions : "possede"

    employees {
        int id PK 
        int id_employee
        int age
        string genre
        int revenu_mensuel
        string statut_marital
        string poste
        int nombre_experiences_precedentes
        int annees_dans_le_poste_actuel
        int nombre_participation_pee
        int nb_formations_suivies
        int niveau_education
        string domaine_etude
        int satisfaction_employee_environnement
        int satisfaction_employee_nature_travail
        int satisfaction_employee_equipe
        int satisfaction_employee_equilibre_pro_perso
        int note_evaluation_actuelle
        string augmentation_salaire_precedente
        int annees_dans_l_entreprise
        int annees_depuis_la_derniere_promotion
        int distance_domicile_travail
        string heure_supplementaires
        int note_evaluation_precedente
        string frequence_deplacement
        int stagnation_promotion
        int charge_trajet_travail
        int evolution
        int heures_sup_bin
        int frequence_deplacement_ord
    }

    predictions {
        int id PK
        int employee_id FK
        string classe_predite
        float probabilite_non
        float probabilite_oui
    }
```
La modification de la base de données dépend exclusivement du développeur car la gestion du SQL est un processus qui lui est dédié.

## 10. Axes d'amélioration

1. Permettre d'ajouter une interface web interactive  avec Gradio ou Streamlit.
2. Généralisation des tests pour protéger davantage le code et atteindre une couverture de 80%.
3. Migration sur RENDER possible.

