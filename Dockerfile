# syntax=docker/dockerfile:1

# =============================================================================
# Étape 1 — CONSTRUCTION
# On installe ici tout ce qui est volumineux (uv, compilateurs éventuels, cache
# des roues) pour que rien de tout cela ne finisse dans l'image livrée.
# =============================================================================
FROM python:3.13-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# uv est installé depuis PyPI plutôt que copié depuis son image officielle :
# une version de moins à suivre, et le résultat est identique.
RUN pip install --no-cache-dir uv

WORKDIR /app

# Couche de dépendances, isolée volontairement : elle n'est reconstruite que
# si pyproject.toml ou uv.lock changent. Modifier le code de l'API ne
# réinstalle donc pas scikit-learn à chaque build.
# --frozen : uv refuse de mettre le lock à jour, ce qui garantit que l'image
# contient exactement les versions testées par la CI.
# --no-dev : ni pytest ni ruff dans l'image de production.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Puis le projet lui-même (layout src/, déclaré dans pyproject.toml).
# README.md est requis par le backend de build : sans lui, uv sync échoue.
COPY README.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev


# =============================================================================
# Étape 2 — EXÉCUTION
# Image finale : l'environnement virtuel construit à l'étape 1, le code
# applicatif, et rien d'autre. Ni uv, ni cache pip, ni outils de compilation.
# =============================================================================
FROM python:3.13-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# libpq : bibliothèque cliente PostgreSQL dont psycopg a besoin à l'exécution.
# Elle est absente de python:3.13-slim, d'où l'erreur « no pq wrapper available »
# si on l'oublie. libpq5 est la bibliothèque seule (quelques Mo) ; libpq-dev,
# qui sert à compiler, n'a rien à faire dans une image de production.
RUN apt-get update \
    && apt-get install --no-install-recommends --yes libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Utilisateur non privilégié : si l'API est compromise, l'attaquant n'est pas
# root dans le conteneur.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Seuls les modules nécessaires au service. Les tests, le notebook et les CSV
# restent dehors (voir .dockerignore).
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser database/ ./database/
COPY --chown=appuser:appuser data/ ./data/
COPY --chown=appuser:appuser models/ ./models/

USER appuser

EXPOSE 8000

# Le port est lu dans la variable PORT, avec 8000 par défaut.
# C'est ce qui permet à la même image de tourner en local sous compose ET sur
# une plate-forme comme Render, qui impose son propre port à l'exécution.
# La forme ["sh", "-c", ...] est nécessaire : sans shell, ${PORT} ne serait
# pas substitué et uvicorn recevrait la chaîne littérale.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]