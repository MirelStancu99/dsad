# Clustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

print(help(AgglomerativeClustering))

# cluster = AgglomerativeClustering(n_clusters=5)
# cluster.fit()
# matrice ierarhica
# m = linkage()

# dendograma

# dendrogram()
# plt.show()

#scor silueta
# silhouette_score()

# factor analysis
import factor_analyzer
from factor_analyzer import FactorAnalyzer

# test_bartlett = factor_analyzer.calculate_bartlett_sphericity()
# kmo = factor_analyzer.calculate_kmo()
# fa = FactorAnalyzer(n_factors= , rotation=)
# fa.fit()

# ev, v = fa.get_eigenvalues()
# ev contains eigenvalues
# usefull for Kaiser criterion (if eigenvalue >= 0 then that factor is relevant)
#
# matricea de corelatie
# fa.loadings_
#
# scoruri
# fa.transform()
#
# comunalitati
# fa.get_communalities()


# PCA

from sklearn.decomposition import PCA
# pca = PCA(n_components=).fit()

# varianta
# alpha = pca.explained_variance_

# varianta Percentage of variance explained by each of the selected components
# alpha = pca.explained_variance_ratio_

# scoruri
# s = pca.transform()

# corelatii
# c = np.corcoef(dataset, scoruri, rowvar=False)

# LDA
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# divizare in set de invatare si set de test
# x_train, y_train, x_test, y_test = train_test_split(predictori, tinta, 0.4 )

# lda = LinearDiscriminantAnalysis().fit(x_train, y_train)

# scoruri
# s = lda.transform(x_test)

#testare

# lda.predict()