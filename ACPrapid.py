import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

t_tari = pd.read_csv('./data_IN/MiseNatPopTari.csv', index_col = 0)
t_continente = pd.read_csv('./data_IN/CoduriTariExtins.csv', index_col = 0)

'''
Salveaza tarile in care rata sporului natural este negativa
'''
cerinta1 = t_tari[t_tari['RS']<0]

cerinta1.to_csv('./data_OUT/cerinta1ACP.csv')

'''
Valorile medii pentru indicatorii de mai sus la nivel de continent
'''

t1 = t_tari.merge(right=t_continente,left_index=True,right_index=True)

indicatori = list(t_tari.columns[2:].values)

cerinta2 = t1[indicatori + ['Continent_Name']].groupby(by='Continent_Name').agg('mean')

cerinta2.to_csv('./data_OUT/cerinta2ACP.csv')

'''
Variantele componentelor principale
'''
randuri = t_tari.index.values
coloane = t_tari.columns.values[2:]

X = t_tari[coloane].values


def standardizare(X):
    medii = np.mean(X, axis = 0)
    abateri = np.std(X, axis = 0)
    return (X - medii) / abateri

Xstd = standardizare(X)

n, m =np.shape(Xstd)

#MODEL PCA
n_componente = min(n,m)
pca = PCA(n_components=n_componente)
pca.fit(Xstd)

#Varianta
alpha = pca.explained_variance_
print(alpha,sum(alpha))

#Calcul scoruri

scoruri = pca.transform(Xstd)
s = pd.DataFrame(scoruri)

#Corelatii
c = np.corrcoef(Xstd,scoruri,rowvar=False)[:m,m:]
c_df = pd.DataFrame(c)

