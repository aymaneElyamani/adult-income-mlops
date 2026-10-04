"""Recupere le jeu de donnees Adult Income et prepare un echantillon de test.

Si le reseau n'est pas disponible (salle isolee), un jeu synthetique de meme
schema est genere : le TP reste realisable hors ligne.
"""

import argparse
import os
import random

COLONNES = ["age", "workclass", "education_num", "marital_status", "occupation",
            "relationship", "sex", "capital_gain", "capital_loss",
            "hours_per_week", "income"]

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "adult/adult.data")

BRUT = ["age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income"]


def depuis_reseau(sortie):
    import pandas as pd
    df = pd.read_csv(URL, header=None, names=BRUT,
                     skipinitialspace=True, na_values="?")
    df["income"] = df["income"].str.replace(".", "", regex=False)
    df[COLONNES].to_csv(sortie, index=False)
    return len(df)


def synthetique(sortie, n=32000, seed=42):
    import csv
    random.seed(seed)
    workclass = ["Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Federal-gov"]
    marital = ["Never-married", "Married-civ-spouse", "Divorced", "Widowed"]
    occupation = ["Tech-support", "Craft-repair", "Sales", "Exec-managerial",
                  "Prof-specialty", "Other-service", "Machine-op-inspct"]
    relationship = ["Husband", "Wife", "Own-child", "Not-in-family", "Unmarried"]

    with open(sortie, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLONNES)
        for _ in range(n):
            age = random.randint(17, 80)
            edu = random.randint(1, 16)
            heures = max(1, min(99, int(random.gauss(40, 12))))
            gain = 0 if random.random() < 0.92 else random.randint(100, 20000)
            perte = 0 if random.random() < 0.95 else random.randint(100, 3000)
            statut = random.choice(marital)
            metier = random.choice(occupation)
            # revenu correle a l'education, aux heures, au capital et au statut :
            # le signal est volontairement net pour que le modele soit apprenable
            score = (0.34 * edu + 0.055 * heures + 0.00035 * gain
                     + 0.028 * age
                     + (1.6 if statut == "Married-civ-spouse" else 0.0)
                     + (1.1 if metier in ("Exec-managerial", "Prof-specialty") else 0.0)
                     + random.gauss(0, 0.9))
            proba = 1 / (1 + pow(2.718, -(score - 7.4)))
            revenu = ">50K" if random.random() < proba else "<=50K"
            w.writerow([age, random.choice(workclass), edu,
                        statut, metier,
                        random.choice(relationship),
                        random.choice(["Male", "Female"]),
                        gain, perte, heures, revenu])
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/adult.csv")
    ap.add_argument("--hors-ligne", action="store_true",
                    help="force la generation synthetique")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.hors_ligne:
        n = synthetique(args.out)
        print("Jeu synthetique genere : %s (%d lignes)" % (args.out, n))
    else:
        try:
            n = depuis_reseau(args.out)
            print("Jeu telecharge : %s (%d lignes)" % (args.out, n))
        except Exception as e:
            print("Telechargement impossible (%s) -- bascule en synthetique." % e)
            n = synthetique(args.out)
            print("Jeu synthetique genere : %s (%d lignes)" % (args.out, n))

    # echantillon de test pour predict.py
    import csv
    with open(args.out, newline="", encoding="utf-8") as f:
        lignes = list(csv.reader(f))
    with open("data/sample.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(lignes[:51])
    print("Echantillon ecrit : data/sample.csv (50 lignes)")


if __name__ == "__main__":
    main()
