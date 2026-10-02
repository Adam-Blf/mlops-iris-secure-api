"""Etapes 3 et 4.2 : verifier que /predict repond aux bons UUID et renvoie 403 aux autres.

Lancer : python -m pytest -v
(train_model.py doit avoir ete lance avant, pour que model.joblib existe.)

TestClient appelle l'application directement, sans lancer uvicorn :
les tests ne dependent d'aucun port ni d'aucun serveur deja demarre.
"""
import pytest
from fastapi.testclient import TestClient

from app import app

VALID_UUID = "3f8b2c1e-7a4d-4e5f-9b6a-1c2d3e4f5a6b"   # present dans users.json
UNKNOWN_UUID = "00000000-0000-4000-8000-000000000000"  # bon format, absent de la base

# Une fleur setosa typique (premiere ligne du dataset iris).
SETOSA = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


@pytest.fixture(scope="module")
def client():
    # Le "with" declenche le lifespan : le modele et users.json sont charges.
    with TestClient(app) as c:
        yield c


def test_uuid_valide_renvoie_une_prediction(client):
    r = client.post("/predict", json=SETOSA, headers={"Authorization": VALID_UUID})
    assert r.status_code == 200
    assert r.json()["species"] == "setosa"


def test_format_bearer_accepte(client):
    r = client.post("/predict", json=SETOSA, headers={"Authorization": f"Bearer {VALID_UUID}"})
    assert r.status_code == 200


def test_uuid_en_majuscules_accepte(client):
    # Un UUID ne depend pas de la casse : uuid.UUID() normalise.
    r = client.post("/predict", json=SETOSA, headers={"Authorization": VALID_UUID.upper()})
    assert r.status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        {},                                   # pas d'en-tete du tout
        {"Authorization": ""},                # en-tete vide
        {"Authorization": UNKNOWN_UUID},      # UUID valide mais inconnu
        {"Authorization": "pas-un-uuid"},     # n'importe quoi
    ],
    ids=["sans-entete", "entete-vide", "uuid-inconnu", "format-invalide"],
)
def test_acces_refuse_en_403(client, headers):
    r = client.post("/predict", json=SETOSA, headers=headers)
    assert r.status_code == 403
    assert r.json() == {"detail": "Access denied"}


def test_corps_invalide_renvoie_422(client):
    # Authentifie mais requete mal formee : c'est Pydantic qui refuse, pas l'auth.
    r = client.post("/predict", json={"sepal_length": "abc"}, headers={"Authorization": VALID_UUID})
    assert r.status_code == 422


def test_health_reste_public(client):
    assert client.get("/health").json() == {"status": "ok"}
