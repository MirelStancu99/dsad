import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as hclast

tabel_alcool = pd.read_csv('data_IN/alcohol.csv',index_col=0)
tabel_continente = pd.read_csv('data_IN/CoduriTariExtins.csv',index_col=0)

tabel = tabel_alcool.merge(right=tabel_continente,left_index=True,right_index=True)
# media consumului pe 5 ani pt fiecare tara, Cod si Medie

ani = list(tabel_alcool.columns[1:].values)

def medieAni(t):
    x=t.values
    medie = np.mean(x)
    return pd.Series(data=medie,  index=['Medie'])

tabel1 = tabel_alcool[['Code']+ ani].groupby(by='Code').agg(sum).apply(
    func=medieAni,axis=1)
print(tabel1)

# Cerinta 2 => Anul in care s-a inregistrat cea mai mare valoare medie
# a consumului la nivel de continent: nume_continent, an

def maximMediuContinent(t):
    x=t.values
    max=np.argmax(x)
    return pd.Series(data=t.index[max],index=['An'])

tabel2 = tabel[['Continent_Name']+ ani].groupby(by='Continent_Name').agg(np.mean).apply(
func=maximMediuContinent, axis=1)

# Cerinta 3 => Analiza ward (cluster)
x = tabel_alcool[ani].values
n, m = x.shape

metoda = 'ward'
# calc matrice
h = hclast.linkage(x, method=metoda)
print(h)