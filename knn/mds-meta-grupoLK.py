import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def atributos_MA(data):
    # Defining X variables for the input of SOM
    X = data.iloc[:, 2:12].values
    # X variables:
    print(pd.DataFrame(X))
    #sc = MinMaxScaler(feature_range = (0, 1))
    #X = sc.fit_transform(X)
    #print(X)
    return X

#X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]
#print(X)

X = atributos_MA(pd.read_csv('/home/leo/projetos/knn/base_conhecimento.csv',sep=','))
print(X)

Z = linkage(X, method='single', metric='euclidean')
print(Z)
#fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)
plt.show()

#Z = pd.DataFrame(Z)
X = pd.DataFrame(X)

# Compute the correlation matrix
corr = X.corr(method='pearson')

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(corr, vmin=-1, vmax=1, center=0, annot=True, cmap='vlag')

# Draw the heatmap with the mask and correct aspect ratio
#sns.heatmap(corr)#, mask=mask, cmap=cmap, center=0,
            #square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()