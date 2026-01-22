import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# nous lisons le fichier csv
path=r"C:\Users\couea\Downloads\finance\Detection_fraud\Input_projet_LVMH.csv"

data= pd.read_csv(path,sep=";")
# nous affichons les types de données et les valeurs manquantes
data.info()
print(data.isnull().sum())

data.boxplot()
plt.show()

