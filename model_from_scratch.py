#%% modèle from scratch (return to 0)
import pandas as pd
import numpy as np

# ============================================================
# 1. Chargement des données
# ============================================================

path = r"./csv/Input_projet_LVMH.csv"
data = pd.read_csv(path, sep=";")

# Conversion de la colonne date en format datetime
data["date"] = pd.to_datetime(data["date"], format="%d/%m/%Y")

# Création d’une variable temporelle "trimestre"
data["quarter"] = data["date"].dt.to_period("Q")

# ============================================================
# 2. Nettoyage et préparation des données
# ============================================================

# Conversion de la colonne de clôture en float
# (gestion du séparateur décimal ",")
data["clot"] = (
    data["clot"]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

# Tri chronologique (important pour les séries temporelles)
data = data.sort_values("date").reset_index(drop=True)

# Calcul du rendement journalier
data["ret"] = data["clot"].pct_change()

# Calcul de la volatilité glissante sur 20 jours
data["vol20"] = data["ret"].rolling(20).std()

# Extraction de l’année (utile pour analyses temporelles)
data["year"] = data["date"].dt.year

# Suppression des lignes avec valeurs manquantes
df = data.dropna(subset=["ret", "vol20"]).copy()

# ============================================================
# 3. Fonctions pour le modèle de détection d’anomalies
#    (Gaussienne multivariée)
# ============================================================

def estimate_gaussian(X):
    """
    Estime les paramètres (moyenne et variance)
    d’une distribution gaussienne multivariée.
    """
    mu = np.mean(X, axis=0)
    var = np.mean((X - mu) ** 2, axis=0)
    return mu, var


def multivariate_gaussian(X, mu, var):
    """
    Calcule la densité de probabilité
    d’une loi gaussienne multivariée.
    """
    k = len(mu)

    # Sécurité numérique pour éviter division par zéro
    var = np.where(var == 0, 1e-12, var)

    norm = (2 * np.pi) ** (-k / 2) * np.prod(var) ** (-0.5)
    exp_term = np.exp(-0.5 * np.sum(((X - mu) ** 2) / var, axis=1))
    return norm * exp_term

# ============================================================
# 4. Détection d’anomalies par trimestre
# ============================================================

results = []

# Le modèle est entraîné indépendamment pour chaque trimestre
for quarter, g in df.groupby("quarter"):

    # Variables utilisées pour la détection d’anomalies
    X = g[["ret", "vol20"]].values

    # Standardisation des variables (centrage-réduction)
    # afin d’éviter l’influence des échelles
    mu_X = X.mean(axis=0)
    std_X = X.std(axis=0, ddof=0)
    Xn = (X - mu_X) / std_X

    # Estimation des paramètres de la gaussienne
    mu, var = estimate_gaussian(Xn)

    # Calcul de la probabilité de chaque observation
    p = multivariate_gaussian(Xn, mu, var)

    # Seuil d’anomalie : 1% des observations les plus rares
    epsilon = np.percentile(p, 1.0)

    # Sauvegarde des résultats
    g = g.copy()
    g["p"] = p
    g["is_anomaly"] = g["p"] < epsilon
    g["epsilon"] = epsilon

    results.append(g)

# Fusion de tous les trimestres
df_quarterly = pd.concat(results)

# ============================================================
# 5. Post-traitement des anomalies
# ============================================================

# Extraction des anomalies strictes
anomalies = df_quarterly[df_quarterly["is_anomaly"]]

# Tri chronologique final
df_quarterly = df_quarterly.sort_values("date")

# Extension des anomalies :
# on marque également le jour suivant comme suspect
df_quarterly["is_anomaly_extended"] = (
    df_quarterly["is_anomaly"] |
    df_quarterly["is_anomaly"].shift(-1)
)

# ============================================================
# 6. Affichage des résultats
# ============================================================

print("Nombre d'anomalies detectées :", len(anomalies))
print(
    anomalies[["date", "quarter", "clot", "ret", "vol20", "p"]].head(20)
)

# Affichage complet du tableau
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# Tableau des jours et des lendemains des anomalies detectées
print("Nombre d'anomalies detectées avec retard d'un jour :", len(df_quarterly[df_quarterly["is_anomaly_extended"]]))
display(
    df_quarterly[df_quarterly["is_anomaly_extended"]]
)
# %%
