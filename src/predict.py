"""Service d'inference : API HTTP minimale autour du modele entraine.

Deux modes :
    python -m src.predict --input data/sample.csv     # lot, en ligne de commande
    uvicorn src.predict:app --host 0.0.0.0 --port 8000  # service HTTP
"""

import argparse
import os
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.common import COLONNES_CAT, COLONNES_NUM

MODEL_PATH = os.environ.get("MODEL_PATH", "/artifacts/model.joblib")

app = FastAPI(title="Adult Income - service d'inference", version="1.0.0")
_modele = None
_charge_le = None


def charger_modele():
    """Chargement paresseux : le modele n'est lu qu'une fois, au premier appel.

    En production on prefere le charger au demarrage (evenement lifespan) pour
    que la premiere requete ne paie pas le cout. Voir seance 8.
    """
    global _modele, _charge_le
    if _modele is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Modele introuvable : %s" % MODEL_PATH)
        t0 = time.time()
        _modele = joblib.load(MODEL_PATH)
        _charge_le = time.time() - t0
        print("Modele charge en %.2f s" % _charge_le, flush=True)
    return _modele


class Individu(BaseModel):
    age: int = Field(..., ge=17, le=100)
    education_num: int = Field(..., ge=1, le=16)
    hours_per_week: int = Field(..., ge=1, le=99)
    capital_gain: int = 0
    capital_loss: int = 0
    workclass: str = "Private"
    marital_status: str = "Never-married"
    occupation: str = "Tech-support"
    relationship: str = "Not-in-family"
    sex: str = "Male"


@app.get("/health")
def sante():
    """Sonde de vivacite : le processus repond-il ?"""
    return {"statut": "ok"}


@app.get("/ready")
def pret():
    """Sonde de disponibilite : le modele est-il charge et utilisable ?

    La distinction health/ready est essentielle : elle devient
    livenessProbe / readinessProbe en Kubernetes (seance 3).
    """
    try:
        charger_modele()
        return {"statut": "pret", "modele": MODEL_PATH,
                "duree_chargement_s": round(_charge_le or 0, 3)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/v1/predict")
def predire(individu: Individu):
    modele = charger_modele()
    df = pd.DataFrame([individu.model_dump()])
    t0 = time.time()
    proba = float(modele.predict_proba(df)[0, 1])
    return {
        "probabilite_revenu_superieur_50k": round(proba, 4),
        "prediction": ">50K" if proba >= 0.5 else "<=50K",
        "latence_ms": round((time.time() - t0) * 1000, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    modele = joblib.load(args.model)
    df = pd.read_csv(args.input)[COLONNES_NUM + COLONNES_CAT]
    proba = modele.predict_proba(df)[:, 1]
    for i, p in enumerate(proba[:10]):
        print("ligne %3d : %.4f  ->  %s" % (i, p, ">50K" if p >= 0.5 else "<=50K"))
    print("%d lignes traitees" % len(df))


if __name__ == "__main__":
    main()
