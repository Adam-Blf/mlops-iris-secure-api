"""Etapes 3 et 4 : le modele iris servi par FastAPI, protege par un UUID.

Lancer :  uvicorn app:app --reload --port 8000
Doc auto : http://127.0.0.1:8000/docs

Appel valide (un UUID present dans users.json, dans l'en-tete Authorization) :
  curl -X POST http://127.0.0.1:8000/predict ^
       -H "Authorization: 3f8b2c1e-7a4d-4e5f-9b6a-1c2d3e4f5a6b" ^
       -H "Content-Type: application/json" ^
       -d "{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}"
"""
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

HERE = Path(__file__).parent
MODEL_PATH = HERE / "model.joblib"
USERS_PATH = HERE / "users.json"

# Ordre des colonnes vu par le modele a l'entrainement. On construit toujours
# l'entree dans cet ordre : si on inversait deux colonnes, scikit-learn ne
# planterait pas, il renverrait juste une prediction fausse qui a l'air normale.
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Etat partage de l'application, rempli une seule fois au demarrage.
state: dict = {}


def load_user_ids(path: Path) -> set[str]:
    """Lit users.json et renvoie l'ensemble des UUID autorises, en minuscules.

    Un set rend la verification "cet UUID existe-t-il ?" instantanee,
    meme avec des milliers d'utilisateurs.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(uuid.UUID(user["uuid"])) for user in data["users"]}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Le modele est charge UNE fois au demarrage du serveur, pas a chaque requete :
    # recharger un fichier a chaque appel ralentirait tout le service.
    bundle = joblib.load(MODEL_PATH)
    state["model"] = bundle["model"]
    state["target_names"] = bundle["target_names"]
    state["user_ids"] = load_user_ids(USERS_PATH)
    yield
    state.clear()


app = FastAPI(title="Iris API securisee", lifespan=lifespan)


class IrisFeatures(BaseModel):
    """Corps JSON attendu par /predict. Pydantic refuse tout seul (code 422)
    un champ manquant, un texte a la place d'un nombre, ou une valeur <= 0."""

    sepal_length: float = Field(gt=0, description="Longueur du sepale en cm")
    sepal_width: float = Field(gt=0, description="Largeur du sepale en cm")
    petal_length: float = Field(gt=0, description="Longueur du petale en cm")
    petal_width: float = Field(gt=0, description="Largeur du petale en cm")


def require_known_user(authorization: str | None = Header(default=None)) -> str:
    """Dependance FastAPI : s'execute AVANT la route et bloque si l'appelant
    n'est pas dans la base.

    Header(default=None) dit a FastAPI d'aller lire l'en-tete HTTP
    "Authorization" (le nom du parametre donne le nom de l'en-tete).

    Pourquoi 403 et pas 401 : l'enonce demande 403 Forbidden ("access denied").
    En toute rigueur HTTP, 401 veut dire "tu ne t'es pas identifie" et doit
    s'accompagner d'un en-tete WWW-Authenticate, 403 veut dire "je sais qui tu
    es, mais tu n'as pas le droit". Ici on renvoie 403 dans tous les cas refuses.
    """
    if not authorization:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Access denied")

    # On accepte l'UUID nu ("3f8b...") ou au format standard "Bearer 3f8b...".
    token = authorization.removeprefix("Bearer ").strip()

    # uuid.UUID() verifie le format et normalise (majuscules -> minuscules).
    # Une chaine qui n'est pas un UUID est refusee avec le meme 403 : on ne dit
    # pas a l'attaquant SI c'est le format ou l'identifiant qui pose probleme.
    try:
        user_id = str(uuid.UUID(token))
    except ValueError:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Access denied")

    if user_id not in state["user_ids"]:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Access denied")
    return user_id


@app.get("/health")
def health() -> dict:
    """Route publique, sans authentification : sert juste a savoir si le
    serveur tourne (utile pour Docker ou un load balancer)."""
    return {"status": "ok"}


@app.post("/predict")
def predict(features: IrisFeatures, user_id: str = Depends(require_known_user)) -> dict:
    """Renvoie l'espece predite. Depends(...) garantit qu'on n'arrive ici
    qu'avec un UUID valide : sinon la dependance a deja renvoye le 403."""
    # Une ligne = une fleur, valeurs rangees dans l'ordre de FEATURES (voir plus haut).
    # Le modele a ete entraine sur un tableau sans noms de colonnes (iris.data),
    # c'est donc l'ORDRE qui compte ici, d'ou la liste FEATURES.
    row = [[getattr(features, name) for name in FEATURES]]
    model = state["model"]
    class_id = int(model.predict(row)[0])
    probas = model.predict_proba(row)[0]
    return {
        "y_pred": class_id,
        "species": state["target_names"][class_id],
        "confidence": round(float(probas[class_id]), 4),
        "user": user_id,
    }
