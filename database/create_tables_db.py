from connexion_db import engine
from models_tables import Base

Base.metadata.create_all(engine)