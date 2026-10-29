from connexion_db import engine
from models_tables import Base

Base.metadata.create_all(engine)

# # 1. recréer la base (côté PostgreSQL, pas côté Python)
# createdb -h localhost -p 5432 -U postgres attrition
# # ou, si createdb n'est pas dans le PATH :
# psql -h localhost -p 5432 -U postgres -c "CREATE DATABASE attrition;"

# # 2. créer les tables
# uv run python database/create_tables_db.py

# # 3. recharger les données
# uv run python database/insert_raws.py