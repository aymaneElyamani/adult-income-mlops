"""Entraine un classifieur de revenus et serialise le pipeline complet.

Usage :
    python -m src.train --out artifacts/model.joblib
"""

import argparse
import json
import os
import time

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.common import CIBLE, construire_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/adult.csv")
    ap.add_argument("--out", default="/artifacts/model.joblib")
    ap.add_argument("--n-estimators", type=int,
                    default=int(os.environ.get("N_ESTIMATORS", 200)))
    ap.add_argument("--seed", type=int,
                    default=int(os.environ.get("RANDOM_SEED", 42)))
    args = ap.parse_args()

    print("Lecture de %s" % args.data, flush=True)
    df = pd.read_csv(args.data)
    y = (df[CIBLE] == ">50K").astype(int)
    X = df.drop(columns=[CIBLE])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y)

    pipe = construire_pipeline(args.n_estimators, args.seed)
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    duree = time.time() - t0

    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metriques = {
        "exactitude": round(accuracy_score(y_te, pred), 4),
        "f1": round(f1_score(y_te, pred), 4),
        "auc": round(roc_auc_score(y_te, proba), 4),
        "duree_entrainement_s": round(duree, 2),
        "n_entrainement": len(X_tr),
        "graine": args.seed,
    }
    print(json.dumps(metriques, indent=2), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(pipe, args.out)
    with open(os.path.join(os.path.dirname(args.out), "metrics.json"), "w") as f:
        json.dump(metriques, f, indent=2)
    print("Modele ecrit : %s (%.1f Mo)"
          % (args.out, os.path.getsize(args.out) / 1e6), flush=True)


if __name__ == "__main__":
    main()
