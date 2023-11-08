import numpy as np
import pandas as pd
import ranking as rk

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score # Cross Validation Function.
from sklearn.model_selection import KFold # KFold Class.
from sklearn.model_selection import LeaveOneOut
from scipy import stats
from time import perf_counter

def statistic(x):  # permute only `x`
    return stats.spearmanr(x, ranking_padrao.squeeze()).statistic

def knn_mds(df_base_con, k, novo_dataset):
    vetor = base_conhecimento_atributos(df_base_con).values #df_base_con.values
    #vetor = df_base_con_meta_atrib.values
    #print(vetor)

    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(vetor)

    #d = np.random.uniform(0.,1.,size=(1,29))
    #print(novo_dataset)
    neigh_dist, neigh_ind = neigh.kneighbors(novo_dataset)
    #print(neigh_ind.ravel())

    df_rk = pd.DataFrame()
    for id in np.sort(neigh_ind.ravel()):
        #print(df_base.iloc[id,[31,32,33,34,35,36,37,38]])
        l = df_base_con.iloc[id,[31,32,33,34,35,36,37,38]].astype(int)
        df_rk =  pd.concat([df_rk,l],axis=1)#,ignore_index=True) #rdf.append(l,ignore_index=True)
    #teste = pd.DataFrame(rdf.T)
    #print(teste.dtypes)
    #print(rdf.T)
    return rk.ranking_predito(df_rk.T)

#Validação através do coeficiente de Spearman
# Validaçao 1 : Correlação de Spearman valor de 1 representa concordância perfeita 
#               -1 representa discordância total e 0 significa que não estão relacionadas.
def validacao_Spearman(rk_predito, rk_padrao):
    #print('Coeficiente de Correlação Spearman: ',stats.spearmanr(ranking_predito.squeeze(), ranking_padrao.squeeze())[0])

    #res_exact = stats.permutation_test((ranking_predito.squeeze(),), statistic, permutation_type='pairings')
    res_asymptotic = stats.spearmanr(ranking_predito.squeeze(), ranking_padrao.squeeze())
    res_exact = res_asymptotic.pvalue
    #print('Probabilidade de ser casual: ', res_exact.pvalue) #, res_asymptotic.pvalue)  # asymptotic pvalue is too low

    #return stats.spearmanr(ranking_predito.squeeze(), ranking_padrao.squeeze())[0]
    return res_asymptotic.statistic ,res_asymptotic.pvalue, res_exact #res_exact.pvalue

# retorna a base de conhecimento com a parte dos rankings
def base_conhecimento_ranking(bc):
    return bc[['IDMAP', 'IPCA', 'LAMP', 'LMDS', 'MDS', 'PBC', 'TSNE', 'UMAP']].astype(float) #df_base_con[[31,32,33,34,35,36,37,38]]

# separa da base de conhecimento a parte dos meta-atributos
def base_conhecimento_atributos(bc):
    df_base_con_meta_atrib = bc.drop(columns=['BD','MA1','IDMAP', 'IPCA', 'LAMP', 'LMDS', 'MDS', 'PBC', 'TSNE', 'UMAP']) #columns=[0,1,31,32,33,34,35,36,37,38])
    df_base_con_meta_atrib.reset_index(inplace=True, drop=True)
    # ajustando colunas de instancia e dimensão
    df_base_con_meta_atrib['MA2'] = pd.to_numeric(df_base_con_meta_atrib['MA2'])
    df_base_con_meta_atrib['MA3'] = pd.to_numeric(df_base_con_meta_atrib['MA3'])
    #print(df_base_meta)

    # colocando as medidas de nr de instâncias e de dimensões na mesma escala para melhorar a acurácia do Knn
    colunas = ['MA2','MA3']
    scaler = MinMaxScaler()
    df_base_con_meta_atrib[colunas] = scaler.fit_transform(df_base_con_meta_atrib[colunas].astype('float32'))
    #print(df_base_con_meta_atrib)
    return df_base_con_meta_atrib


# carregando a base de conhecimento
df_base_con = pd.read_csv('/home/leo/projetos/knn/base_conhecimento.csv',sep=',')#, header=None)
#print(df_base_con)

ranking_padrao = rk.ranking_padrao(base_conhecimento_ranking(df_base_con))
#print(ranking_padrao)

ds_teste = pd.read_csv('/home/leo/projetos/knn/base_conhecimento_retirados _para_teste.csv',sep=',')#, header=None)
#ds_teste = np.random.uniform(0.,1.,size=(1,29))
#print(ds_teste)
#ds_t_atributos = base_conhecimento_atributos(ds_teste)
#print(ds_t_atributos)
#ds_t_ranking = base_conhecimento_ranking(ds_teste)
#print(ds_t_ranking)
#ds = pd.DataFrame(ds_t_atributos.iloc[0].values).T
#print(ds.values)
#print(np.random.uniform(0.,1.,size=(1,29)))

t_inicio = perf_counter()

k_values = [i for i in range (1,2)]
scores = []
for k in k_values:
    #ranking_predito = knn_mds(df_base_con, k, ds.values)
    #score = cross_val_score(knn, X, y, cv=5)

    # leave-one-out
    loo = LeaveOneOut()
    loo.get_n_splits(df_base_con)
    print(loo)
    for i, (train_index, test_index) in enumerate(loo.split(df_base_con)):
        print(f"k {k} Fold {i}:")
        #print(f"  Train: index={train_index}")
        #print(f"  Test:  index={test_index}")
        treino = df_base_con.drop(test_index, axis='index')
        teste = df_base_con.iloc[test_index]
        #print(treino)
        #print(teste)

        ds_t_atributos = base_conhecimento_atributos(teste)
        #print(ds_t_atributos.values)
        ds_t_ranking = base_conhecimento_ranking(teste)
        #print(ds_t_ranking)

        ranking_predito = knn_mds(treino, k, ds_t_atributos.values)
        #print('Predito: ', ranking_predito)
        #ranking_predito = knn_mds(df_base_con, k, ds.values)
        score = stats.spearmanr(ranking_predito.squeeze(), ds_t_ranking.squeeze()) #pd.DataFrame(ds_t_ranking.iloc[0].values).T.squeeze())
        scores.append(score)

print(scores)
print(np.mean(scores, axis= 0))
t_fim = perf_counter()
print("Tempo de processamento Dataset: ",(t_fim-t_inicio), ' s')

ds = np.random.uniform(0.,1.,size=(1,29))
k = 5
ranking_predito = knn_mds(df_base_con, k, ds)

print('Ranking Predito ... ')
print(ranking_predito.T)
print('Ranking Teste ...')
print(pd.DataFrame(ds_t_ranking.iloc[0].values).T)
print('Ranking Padrão ... ')
print(ranking_padrao)
spr = stats.spearmanr(ranking_predito.squeeze(), pd.DataFrame(ds_t_ranking.iloc[0].values).T.squeeze())
print('Coeficiente de Correlação Spearman com ranquink real ', spr.statistic)
spr = stats.spearmanr(ranking_predito.squeeze(), ranking_padrao.squeeze())
print('Coeficiente de Correlação Spearman com ranquink padrão ', spr.statistic)
print('--------------------------------------------------------')

print('-----------------FINALIADO------------------------------')


'''
# Validaçao 1 : Correlação de Spearman valor de 1 representa concordância perfeita 
#               -1 representa discordância total e 0 significa que não estão relacionadas.
#print("Correlação de Spearman valor de 1 representa concordância perfeita \n -1 representa discordância total e 0 significa que não estão relacionadas.")
coef_spearman, res_exact, res_asymptotic = validacao_Spearman(ranking_predito, ranking_padrao)
print('Coeficiente de Correlação Spearman com ranking padrao: ', coef_spearman)
#print('Probabilidade de ser casual: ', res_asymptotic)
#print('Probabilidade de ser casual exata: ', res_exact)

print("Processo finalizado")

# leave-one-out
loo = LeaveOneOut()
loo.get_n_splits(df_base_con_meta_atrib)
print(loo)
for i, (train_index, test_index) in enumerate(loo.split(df_base_con_meta_atrib)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")

'''
'''# K-FOld
kfold  = KFold(n_splits=5, shuffle=True) # shuffle=True, Shuffle (embaralhar) the data.
result = cross_val_score(neigh, df_base_con_meta_atrib, cv = kfold)

print("K-Fold (R^2) Scores: {0}".format(result))
print("Mean R^2 for Cross-Validation K-Fold: {0}".format(result.mean()))
'''

'''
validação cruzada python https://drigols.medium.com/introdu%C3%A7%C3%A3o-a-valida%C3%A7%C3%A3o-cruzada-k-fold-2a6bced32a90
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
'''


'''
# Criando conjunto de dados
#X, y = make_blobs(n_samples = 500, n_features = 30, centers = 7,cluster_std = 1.5, random_state = 4)
X, y = make_blobs(n_samples = 10, n_features = 2, centers = 7,cluster_std = 1.5, random_state = 4)
saidas = []


#print(X)

samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
print(samples)

neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(samples)
print(neigh.kneighbors([[1., 1., 1.]]))

# Visualize o conjunto de dados'''
'''plt.style.use('seaborn')
plt.figure(figsize = (10,10))
plt.scatter(X[:,0], X[:,1], c=y, marker= '*',s=100,edgecolors='black')
plt.show()'''

'''
# Dividindo dados em conjuntos de dados de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
#X_train, X_test = train_test_split(df_base, random_state = 0)

# Implementação do Classificador KNN para k qualquer
k = 3
knn = KNeighborsClassifier(n_neighbors=k)

# Previsões para os Classificadores KNN
knn.fit(X_train, y_train)

# retorna o indice dos vizinhos proximos para uma amostra de entrada
#d = np.random.randint(-10.,10.,size=(1,30))
d = np.random.randint(-10.,10.,size=(1,2))
idx = knn.kneighbors(d, return_distance=False)
#for i in idx:
#    print(X[i])

y_pred = knn.predict(X_test)

# Preveja a precisão para os valores de k
print("Accuracy with k =", k , ":", accuracy_score(y_test, y_pred)*100)

# Contabilizar acertos para os dados de teste
acertos, indice_rotulo = 0, 0
for i in range(len(X_test), len(X_train)):
    if y_pred[indice_rotulo] == y_train[i]:
        acertos += 1
    indice_rotulo += 1
print('Total de treinamento: %d' % X_train)
print('Total de testes: %d' % (len(X_test) - X_train))
print('Total de acertos: %d' % acertos)
print('Porcentagem de acertos: %.2f%%' % (100 * acertos / (len(X_test) - X_train)))
'''
# Visualize previsões
'''
plt.style.use('seaborn')
plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred, marker= '*', s=100,edgecolors='black')
plt.title("Predicted values with k", fontsize=20)
plt.show()

https://www.monolitonimbus.com.br/classificacao-usando-knn/
'''