"""Authentification par clé API partagée (header X-API-Key)."""
import os
import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Vérifie que la clé API fournie dans l'en-tête X-API-Key est valide.

    Utilise secrets.compare_digest plutôt qu'une simple égalité (==) pour se
    protéger contre les attaques par mesure de temps (timing attacks) :
    une comparaison == s'arrête à la première différence de caractère, ce
    qui peut fuiter subtilement des informations sur la clé attendue via le
    temps de réponse. compare_digest compare en temps constant.
    """
    cle_attendue = os.getenv("SECRET_API_KEY")

    if not cle_attendue:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_KEY non configurée côté serveur.",
        )
    if not api_key or not secrets.compare_digest(api_key, cle_attendue):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API invalide ou manquante (en-tête X-API-Key requis).",
        )
    return api_key