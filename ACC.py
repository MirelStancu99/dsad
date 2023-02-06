import pandas as pd
import numpy as np
import sklearn.cross_decomposition as skl

'''
In order to perform canonical correlation analysis, standardize the value of the variables and split
the initial data set into 2 data subsets analysis as follows:
Pork meat production, Beef meat production, Sheep and goat meat production, Poultry production – set X;
Pork meat consumption, Beef consumption, Sheep and goat meat consumption, Poultry consumption – set Y.
Save the 2 standardized data sets in files Xstd.csv and Ystd.csv.
'''


tabel = pd.read_csv('./data_IN/DataSet_34.csv',index_col=0)

obs_nume = tabel.index.values
var_nume = tabel.columns.values

x_coloane = var_nume[:4]
y_coloane = var_nume[4:]
X = tabel[x_coloane].values
Y = tabel[y_coloane].values

#standardizare pe ndarray
def standardizare(X):
    medii=np.mean(X,axis=0)
    abateri=np.std(X,axis=0)
    return (X-medii)/abateri
Xstd = standardizare(X)
Xstd_df = pd.DataFrame(data=Xstd,index=obs_nume,columns=x_coloane)

Ystd = standardizare(Y)
Ystd_df = pd.DataFrame(data=Ystd,index=obs_nume,columns=y_coloane)

Xstd_df.to_csv('./data_OUT/Xstd.csv')

# creare model ACC, calculare scoruri canonice
# n- nr linii, p- nr coloane
# q - nr coloane Y

n, p = np.shape(X)
q = np.shape(Y)[1] #nr coloane
m = min(p,q)

modelACC = skl.CCA(n_components=m)
modelACC.fit(X=Xstd,Y=Ystd)

z, u = modelACC.transform(X=Xstd,Y=Ystd)
#X scores
z_df=pd.DataFrame(data=z, index=obs_nume,
                  columns=['z'+ str(j+1) for j in range(p)])

u_df=pd.DataFrame(data=u, index=obs_nume,
                  columns=['u'+str(j+1) for j in range(p)])
print(u_df)

'''
Determine and save the factor loadings corresponding to variables from 
X and Y data sets in the files Rxz.csv and Ryu.csv respectively 
'''
Rxz = modelACC.y_loadings_
Rxz_df = pd.DataFrame(data=Rxz, index=x_coloane,
                      columns=['z'+ str(j+1) for j in range(p)])

Ryu = modelACC.y_loadings_
Ryu_df = pd.DataFrame(data=Ryu, index=x_coloane,
                      columns=['z'+ str(j+1) for j in range(p)])