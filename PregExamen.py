import pandas as pd
import numpy as np

tabel_industrii = pd.read_csv('./data_IN/Industrie.csv',index_col=0)
tabel_populatie = pd.read_csv('./data_IN/PopulatieLocalitati.csv',index_col=0)

# Cerinta 1 - cifra de afaceri pe nr loc
tabel = tabel_industrii.merge(right=tabel_populatie,right_index=True,left_index=True)

industrii = list(tabel_industrii.columns[1:].values)

def CA(t, variabile, populatie):
    x = t[variabile].values/t[populatie]
    v = list(x)
    v.insert(0,t['Localitate_x'])
    return pd.Series(data=v,index=['Localitate']+ variabile)


t2 = tabel[['Localitate_x', 'Populatie'] + industrii].apply(
    func=CA, axis=1, variabile = industrii, populatie = 'Populatie'
)

# Cerinta 2 - cifra de afaceri maxima per judet
def maxCA(t):
    x = t.values
    max_linie = np.argmax(x)
    return pd.Series(data=[t.index[max_linie], x[max_linie]],
                     index=['Activitate','CifraAfaceri'])
t3 = tabel[industrii+['Judet']].groupby(by='Judet').agg(sum)

t4=t3[industrii].apply(func=maxCA, axis=1)


# Cerinta 3 - Localitatile cu populatia peste 100.000

tabelC3 = tabel_populatie[tabel_populatie['Populatie']>100000]

def medie(t,variabile):
    medie=np.mean(tabel[variabile].values)
    return pd.Series(data=medie,index=['Media'])

#medie industrii pe judete
tabelAlt = tabel[industrii + ['Judet']].groupby(by='Judet').agg('mean')
print(tabelAlt)
