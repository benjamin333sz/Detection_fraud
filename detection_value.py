import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# nous lisons le fichier csv
path=r"C:\Users\benja\Desktop\cours_ESME\cours\INGE\INGE_3\semaine_mineur\détection_valeur_aberrante\Input_projet_LVMH.csv"

data= pd.read_csv(path,sep=";")
# nous affichons les types de données et les valeurs manquantes
data.info()
print(data.isnull().sum())

data.boxplot()
plt.show()

