import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

### Visualisation de la donnée ###

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
data_comparison = data_input.copy()
data_comparison['aberrante'] = data_true['clot'] != data_input['clot']
data['aberrante']= data_true['clot'] != data_input['clot']

# Sauvegarder le résultat dans un nouveau fichier CSV
data_comparison.to_csv('Training_Values.csv', sep=';', index=False)
print("Fichier 'Training_Values.csv' créé avec succès!")

# Trie par date
data = data.sort_values("date").reset_index(drop=True)

# Exemple de features adaptées time series :
# - rendement journalier
# - volatilité rolling (20 jours) du rendement
data["ret"] = data["clot"].pct_change()
data["vol20"] = data["ret"].rolling(20).std()

# Nous prenons également les clots moyennes et volatiles sur 20 jours
data["clot_mean20"] = data["clot"].rolling(20).mean()
data["clot_std20"]  = data["clot"].rolling(20).std()

# conserve les features dans un dataframe à part
df = data.dropna(subset=["ret", "vol20","aberrante","clot_mean20","clot_std20"]).copy()

# ajout des rendements précédent, suivant, et la différence entre les deux
df["ret_prev"] = df["clot"] / df["clot"].shift(1) - 1
df["ret_next"] = df["clot"].shift(-1) / df["clot"] - 1
df["ret_diff"] = abs(df["ret_prev"] - df["ret_next"])

# Z-score local pour savoir s'il y a une anomalie
df["z_clot"] = abs(
    (df["clot"] - df["clot_mean20"]) / df["clot_std20"]
)
df["is_anomaly_rule_z"] = df["z_clot"] > 6

# Regarde la rupture locale avant, après.
# Nous regardons, l'erreur relatif local
df["rel_error_prev"] = abs(df["clot"] - df["clot"].shift(1)) / df["clot"].shift(1)
df["rel_error_next"] = abs(df["clot"] - df["clot"].shift(-1)) / df["clot"]
df["rel_error"] = df[["rel_error_prev", "rel_error_next"]].min(axis=1)
df["is_anomaly_rule"] = df["rel_error"] > 0.15   # 15% de rupture locale


# Prix attendu
df["clot_pred"] = df["clot"].shift(1)
df["residual"] = abs(df["clot"] - df["clot_pred"])


X = df[["rel_error"]].dropna().values

# Standardisation
X_mu = X.mean(axis=0)
X_std = X.std(axis=0, ddof=0)
Xn = (X - X_mu) / X_std

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
# =========================
# Modèle Isolation Forest
# =========================

iso_forest = IsolationForest(
    n_estimators=300,
    contamination=10/len(df),  # 10 anomalies attendues sur l'ensemble des données
    random_state=42
)

# entraînement du modèle
iso_forest.fit(Xn)

# -1 = anomalie, 1 = normal
iso_pred = iso_forest.predict(Xn)

# ajout de prediction anomalies par le modèle
df["is_anomaly_iforest"] = iso_pred == -1


# Résultat pour le isolation forest : nombre d'anomalie puis affichage des anomalies
df["iforest_score"] = iso_forest.decision_function(Xn)
print("Nombre d'anomalies détectées (Isolation Forest) :",
      df["is_anomaly_iforest"].sum())

anomalies_if = df[df["is_anomaly_iforest"]].copy()
print(
    anomalies_if[
        ["date", "clot", "ret", "vol20", "iforest_score"]
    ].sort_values("iforest_score")
)

y_true = df["aberrante"].astype(int)
y_pred = df["is_anomaly_iforest"].astype(int)
print(confusion_matrix(y_true, y_pred))
print("Rapport pour l'erreur relative locale",classification_report(
    y_true,
    df["is_anomaly_rule"].astype(int)
))

print("Rapport pour le Isolation forest",classification_report(y_true, y_pred, digits=4))

# modèle from scratch

# =========================
# 2) Fonctions du estimation gaussien et multivariance gaussien
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
quantile=0.5
epsilon = np.percentile(p, quantile)

df["p"] = p
df["is_anomaly"] = df["p"] < epsilon

anomalies = df[df["is_anomaly"]].copy()

print("epsilon =", epsilon)
print("Nb anomalies détectées :", len(anomalies))

# Afficher les anomalies (dates + prix)
print(anomalies[["date", "clot", "ret", "vol20", "p"]].head(100))