COLONNES_NUM = ["age", "education_num", "hours_per_week", "capital_gain", "capital_loss"]
COLONNES_CAT = ["workclass", "marital_status", "occupation", "relationship", "sex"]
CIBLE = "income"


def construire_pipeline(n_estimators=200, seed=42):
    """Construit le pipeline complet : pretraitement + modele.

    Le pretraitement fait partie du pipeline serialise : le fichier .joblib
    contient donc tout ce qu'il faut pour predire, sans code externe.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numerique = Pipeline([
        ("imputation", SimpleImputer(strategy="median")),
        ("mise_a_echelle", StandardScaler()),
    ])
    categoriel = Pipeline([
        ("imputation", SimpleImputer(strategy="most_frequent")),
        ("encodage", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pretraitement = ColumnTransformer([
        ("num", numerique, COLONNES_NUM),
        ("cat", categoriel, COLONNES_CAT),
    ])

    return Pipeline([
        ("pretraitement", pretraitement),
        ("modele", HistGradientBoostingClassifier(
            max_iter=n_estimators, random_state=seed)),
    ])
