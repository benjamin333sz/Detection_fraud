#%% importation des bibliothèques
# Benjamin SZUREK - Audren COUÉ - Raphaël BARON 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

#%%Visualisation de la base de données

# nous lisons le fichier csv
path=r"./csv/Input_projet_LVMH.csv"

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
#%% Comparaison excels originales et valeurs faussées
# pour calcul de métrique prochain
data_true = pd.read_csv(r"./csv/True_Value.csv", sep=";")
data_input = pd.read_csv(path, sep=";")

# Créer une colonne de comparaison: True si différent, False si égal
data_comparison = data_input.copy()
data_comparison['aberrante'] = data_true['clot'] != data_input['clot']
data['aberrante']= data_true['clot'] != data_input['clot']

# Sauvegarder le résultat dans un nouveau fichier CSV
data_comparison.to_csv('./csv/Training_Values.csv', sep=';', index=False)
print("Fichier 'Training_Values.csv' créé avec succès!")

#%% Ajout de métrique dans la base de données d'origine
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

#%% Création nouvelle base de données df pour calcul
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

#%% Utilisation base de données X avec l'erreur relatif pour le modèle Isolation Forest
X = df[["rel_error"]].dropna().values

# Standardisation
X_mu = X.mean(axis=0)
X_std = X.std(axis=0, ddof=0)
Xn = (X - X_mu) / X_std

#%% Modèle Isolation Forest

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

#%% modèle from scratch (return to 0)
import pandas as pd
import numpy as np

# ============================================================
# 1. Chargement des données
# ============================================================

path = "Input_projet_LVMH.csv"
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