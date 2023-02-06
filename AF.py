import numpy as np
import pandas as pd
import factor_analyzer as fact

t_vot = pd.read_csv('./data_IN/VotBUN.csv',index_col=0)
t_loc = pd.read_csv('./data_IN/Coduri_Localitati.csv',index_col=0)

table = t_vot.merge(right=t_loc,right_index=True,left_index=True)

votanti = list(t_vot.columns[1:].values)

#Categoria de alegatori cu cel mai mic procent de prezenta la vot

def procentMinim(t):
    x=t.values
    min = np.argmin(x)
    return pd.Series(data=t.index[min], index=['Categorie'])


cerinta1 = t_vot[['Localitate'] +votanti].groupby(
    by=['Siruta','Localitate']).agg(sum).apply(func=procentMinim, axis=1)

print(cerinta1)
# Salvare valori medii la nivel de judet

cerinta2 = table[['Judet'] +votanti].groupby(by='Judet').agg('mean')

# Analiza factoriala

x = t_vot[votanti].values
def standardizare(X):
    medii = np.mean(X, axis = 0)
    abateri = np.std(X, axis = 0)
    return (X - medii) / abateri

x=standardizare(x)

n, m = x.shape

barlett_test = fact.calculate_bartlett_sphericity(x)
print("Barlett test: ",barlett_test)

if(barlett_test[1]>0.0001):
    print("Nu exista factori comuni")

#KMO
kmo = fact.calculate_kmo(x)
print("KMO: ",kmo)

#construire model
model_fact = fact.FactorAnalyzer(n_factors=m,rotation=None)
model_fact.fit(x)
#varianta factorilor
alpha = model_fact.get_factor_variance()

#corelatii variabile - factori
loadings = model_fact.loadings_

#scoruri
sf = model_fact.transform(x)

#comunalitati
comm = model_fact.get_communalities()
