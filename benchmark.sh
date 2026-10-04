#!/usr/bin/env bash
# Mesure taille, duree de construction a froid et duree de reconstruction
# apres modification du code. Usage : bash benchmark.sh <etiquette>
set -euo pipefail
TAG="${1:-naif}"
DOCKERFILE="${2:-Dockerfile.serve}"
IMAGE="adult-serve:${TAG}"

echo "=== Mesure pour ${IMAGE} (${DOCKERFILE}) ==="

echo "--- 1. Construction a froid (--no-cache) ---"
T0=$(date +%s)
docker build --no-cache -f "${DOCKERFILE}" -t "${IMAGE}" . >/dev/null
T_FROID=$(( $(date +%s) - T0 ))

echo "--- 2. Reconstruction apres modification d'une ligne de code ---"
echo "# touche par benchmark.sh $(date +%s)" >> src/predict.py
T0=$(date +%s)
docker build -f "${DOCKERFILE}" -t "${IMAGE}" . >/dev/null
T_CHAUD=$(( $(date +%s) - T0 ))
# on retire la ligne ajoutee
sed -i '$ d' src/predict.py

TAILLE=$(docker image inspect "${IMAGE}" --format '{{.Size}}')
TAILLE_MO=$(( TAILLE / 1000000 ))

echo
printf '%-28s %s\n' "Image"                  "${IMAGE}"
printf '%-28s %s Mo\n' "Taille"              "${TAILLE_MO}"
printf '%-28s %s s\n' "Build a froid"        "${T_FROID}"
printf '%-28s %s s\n' "Rebuild apres modif." "${T_CHAUD}"
echo
echo "Reportez ces trois valeurs dans le tableau du README."
