import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Visualisation de la donnée

# nous lisons le fichier csv
path=r"Input_projet_LVMH.csv"

data= pd.read_csv(path,sep=";")
# nous affichons les types de données et les valeurs manquantes
data.info()
print(data.isnull().sum())

# Créer le dossier img s'il n'existe pas
img_dir = Path("img")
img_dir.mkdir(exist_ok=True)

# Convertir la colonne date en datetime
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')

# Extraire l'année
data['year'] = data['date'].dt.year

# Créer les box plots par année
plt.figure(figsize=(14, 6))
data.boxplot(column='clot', by='year', figsize=(14, 6))
plt.suptitle('Box plots des prix de clôture LVMH par année')
plt.xlabel('Année')
plt.ylabel('Prix de clôture')
plt.savefig(img_dir / 'boxplot_par_annee.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Comparer les deux fichiers CSV
data_true = pd.read_csv(r"True_Value.csv", sep=";")
data_input = pd.read_csv(r"Input_projet_LVMH.csv", sep=";")

# Créer une colonne de comparaison: True si différent, False si égal
data_comparison = data_true.copy()
data_comparison['aberrante'] = data_true['clot'] != data_input['clot']

# Sauvegarder le résultat dans un nouveau fichier CSV
data_comparison.to_csv('Training_Values.csv', sep=';', index=False)
print("Fichier 'Training_Values.csv' créé avec succès!")
print(data_comparison.head(10))

# Vérifier les types
print(data.dtypes)

# Convertir la colonne 'clot' en numérique
data["clot"] = (
    data["clot"]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .astype(float)
)
data = data.sort_values("date").reset_index(drop=True)

data["ret"] = data["clot"].pct_change()
data["vol20"] = data["ret"].rolling(20).std()

print(data.dtypes)
data["date"] = pd.to_datetime(data["date"], format="%d/%m/%Y")
data = data.sort_values("date").reset_index(drop=True)

# Exemple de features adaptées time series :
# - rendement journalier
# - volatilité rolling (20 jours) du rendement
data["ret"] = data["clot"].pct_change()
data["vol20"] = data["ret"].rolling(20).std()

df = data.dropna(subset=["ret", "vol20"]).copy()

X = df[["ret", "vol20"]].values

# (Optionnel mais recommandé) Standardisation
X_mu = X.mean(axis=0)
X_std = X.std(axis=0, ddof=0)
Xn = (X - X_mu) / X_std

# =========================
# 2) Fonctions du notebook
# =========================
def estimate_gaussian(X):
    """
    Retourne mu et var (variance par feature) en supposant covariance diagonale.
    """
    m, n = X.shape
    mu = (1 / m) * np.sum(X, axis=0)
    var = (1 / m) * np.sum((X - mu) ** 2, axis=0)
    return mu, var

def multivariate_gaussian(X, mu, var):
    """
    Densité d'une gaussienne multivariée avec covariance diagonale (var).
    Retourne p(x) pour chaque ligne de X.
    """
    k = len(mu)
    # éviter divisions par 0
    var = np.where(var == 0, 1e-12, var)

    norm = (2 * np.pi) ** (-k / 2) * (np.prod(var) ** (-0.5))
    exp_term = np.exp(-0.5 * np.sum(((X - mu) ** 2) / var, axis=1))
    return norm * exp_term

# =========================
# 3) Entraîner + scorer
# =========================
mu, var = estimate_gaussian(Xn)
p = multivariate_gaussian(Xn, mu, var)

# =========================
# 4) Choisir epsilon (non supervisé)
# =========================
# Exemple: on marque comme anomalies les 0.5% points les moins probables
epsilon = np.percentile(p, 0.5)

df["p"] = p
df["is_anomaly"] = df["p"] < epsilon

anomalies = df[df["is_anomaly"]].copy()

print("epsilon =", epsilon)
print("Nb anomalies détectées :", len(anomalies))

# Afficher les anomalies (dates + prix)
print(anomalies[["date", "clot", "ret", "vol20", "p"]].head(20))