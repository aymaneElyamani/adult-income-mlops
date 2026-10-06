# TP 2 — Conteneuriser un modèle et optimiser son image

Cloud Computing & Déploiement IA — Faculté des Sciences Semlalia (FSSM)
Encadrant : Pr. Mohammed AMEKSA — Année universitaire 2025-2026

Rendu complet (PDF, 29 pages) : [`rapport-tp2.pdf`](rapport-tp2.pdf) — source LaTeX : [`rapport-tp2.tex`](rapport-tp2.tex)

## Objectif

Prendre un classifieur de revenus (Adult Income, UCI) et le livrer sous forme d'image Docker de production : entraînement et service séparés, taille d'image minimale mesurée, exécution non-root, sonde de santé, puis analyse de vulnérabilités.

## Résultats du modèle

Entraînement sur 26 048 lignes (graine 42), `HistGradientBoostingClassifier` sérialisé en pipeline :

| Métrique | Valeur |
| --- | --- |
| Exactitude | 0,8773 |
| F1 | 0,7263 |
| AUC | 0,9298 |
| Durée d'entraînement | 8,24 s |

Source : [`artifacts/metrics.json`](artifacts/metrics.json)

## Résultats d'optimisation

Mesures réelles (Docker 28.5.1) reproduites par [`benchmark.sh`](benchmark.sh), detail dans [`mesure-results.txt`](mesure-results.txt).

| Étape | Fichier | Taille | Build à froid | Rebuild | Gain taille |
| --- | --- | --- | --- | --- | --- |
| Naïf | `Dockerfile.serve` | 624 Mo | 112 s | 106 s | — |
| 1. Base `python:3.11-slim` | `Dockerfile.serve.opt1` | 263 Mo | 108 s | 130 s | −361 Mo |
| 2. `COPY requirements.txt` avant `src/` | `Dockerfile.serve.opt2` | 263 Mo | 159 s | 15 s | 0 |
| 3. `.dockerignore` complet | `Dockerfile.serve.opt3` | 263 Mo | 678 s | 5 s | 0 |
| 4. `pip install --no-cache-dir` | `Dockerfile.serve.opt4` | 172 Mo | 154 s | 16 s | −91 Mo |
| 5. Construction multi-étapes | `Dockerfile.serve.opt5` | 176 Mo | 208 s | 18 s | +4 Mo |
| 6. Non-root + `HEALTHCHECK` | `Dockerfile.serve.opt6` | 176 Mo | 253 s | 30 s | 0 |

**Résultat : 624 Mo → 176 Mo (−72 %), rebuild 106 s → 30 s.** Chaque Dockerfile est conservé séparément pour rendre la mesure reproductible pas à pas. L'image finale est `Dockerfile.serve.opt6` (identique à `Dockerfile.serve.opt`).

## Prérequis

- Docker 28.5.1 (avec BuildKit)
- Docker Compose v2 (orchestration)
- Trivy (analyse de vulnérabilités)
- Python 3.11+ uniquement si vous voulez entraîner hors conteneur

## Démarrage rapide

```bash
# 1. Image d'entraînement
docker build -f Dockerfile.train -t adult-train .

# 2. Entraînement : produit artifacts/model.joblib
docker run --rm -v "$PWD/artifacts:/artifacts" adult-train

# 3. Image de service (variante finale)
docker build -f Dockerfile.serve.opt6 -t adult-serve:opt .

# 4. Lancement de l'API
docker run -d --name adult-serve -p 8000:8000 \
  -v "$PWD/artifacts:/artifacts:ro" adult-serve:opt

# 5. Vérification
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 39, "education_num": 13, "hours_per_week": 40}'
```

## API d'inférence

| Route | Méthode | Rôle |
| --- | --- | --- |
| `/health` | GET | Vivacité : le processus répond-il ? |
| `/ready` | GET | Disponibilité : le modèle est-il chargé ? (503 sinon) |
| `/v1/predict` | POST | Prédiction sur un individu |

La distinction `/health` / `/ready` correspond aux `livenessProbe` et `readinessProbe` Kubernetes.

Le même module fonctionne aussi en lot :

```bash
python -m src.predict --input data/sample.csv
```

## Reproduire les mesures

```bash
bash benchmark.sh naif   Dockerfile.serve
bash benchmark.sh opt1   Dockerfile.serve.opt1
bash benchmark.sh opt6   Dockerfile.serve.opt6
```

Le script mesure la taille de l'image, la durée d'un build `--no-cache` puis celle d'un rebuild après modification d'une ligne de `src/predict.py` (la ligne est retirée automatiquement).

## Analyse de vulnérabilités

```bash
trivy image adult-serve:opt
```

Rapport complet : [`trivy.checks.txt`](trivy.checks.txt). À l.exit, 0 vulnérabilité dans les paquets Python et 6 dans `pip` (supprimables avec `--no-cache-dir` et l'absence de `pip` dans l'image finale).

## Structure du dépôt

```
├── src/
│   ├── common.py           # colonnes, pipeline sklearn
│   ├── train.py            # entraînement + metrics.json
│   └── predict.py          # API FastAPI + mode lot
├── data/
│   ├── adult.csv           # jeu de données UCI
│   ├── sample.csv          # échantillon pour l'inférence
│   └── download.py
├── artifacts/
│   ├── model.joblib        # pipeline sérialisé
│   └── metrics.json
├── Dockerfile.train        # image d'entraînement
├── Dockerfile.serve        # image de service naïve (624 Mo)
├── Dockerfile.serve.opt1…6 # une optimisation par fichier
├── benchmark.sh            # script de mesure
├── mesure-results.txt      # résultats bruts des 8 mesures
├── trivy.checks.txt        # rapport Trivy
├── rapport-tp2.tex/.pdf    # rapport LaTeX
├── screens/                # 13 captures d'écran du TP
└── requirements.txt
```

## Dépendances Python

`scikit-learn==1.5.2`, `pandas==2.2.3`, `numpy==2.1.3`, `joblib==1.4.2`, `fastapi==0.115.5`, `uvicorn[standard]==0.32.1`, `pydantic==2.10.3`

## Notes

- Le modèle est monté en volume, jamais copié dans l'image : `Dockerfile.serve` suppose que `/artifacts/model.joblib` existe au lancement.
- L'image finale tourne sous `appuser` (UID 10001) et déclare un `HEALTHCHECK` sur `/health`.
- `pip`, les caches pip et les outils de build du `builder` ne sont pas présents dans l'image finale.