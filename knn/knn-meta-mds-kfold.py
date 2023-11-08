# Referêrncia: https://medium.com/data-hackers/como-criar-k-fold-cross-validation-na-m%C3%A3o-em-python-c0bb06074b6b

import random
import numpy as np
import pandas as pd
import ranking as rk
import os

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

def kfoldcv(indices, k, seed = 42):
    
    size = len(indices)
    subset_size = round(size / k)
    random.Random(seed).shuffle(indices)
    subsets = [indices[x:x+subset_size] for x in range(0, len(indices), subset_size)]
    kfolds = []
    for i in range(k):
        test = subsets[i]
        train = []
        for subset in subsets:
            if subset != test:
                train.append(subset)
        kfolds.append((train,test))
        #print('fold: ', i, ' ', kfolds)
        #print('__________________________________________________________________________________')
        
    return kfolds

def normalizar_base_conhecimento(bc):
    #print(bc)
    bc_atrib = bc.drop(columns=['BD','MA1','IDMAP', 'IPCA', 'LAMP', 'LMDS', 'MDS', 'PBC', 'TSNE', 'UMAP'])
    bc_ranking = bc.drop(columns=['BD','MA1','MA2','MA3','MA4','MA5','MA6','MA7','MA8','MA9','MA10','MA11','MD1','MD2','MD3','MD4',
                                  'MD5','MD6','MD7','MD8','MD9','MD10','MD11','MD12','MD13','MD14','MD15','MD16','MD17','MD18','MD19']) 
    #print(bc_atrib)
    scaler = MinMaxScaler()
    bc_atrib = pd.DataFrame(scaler.fit_transform(bc_atrib.astype('float32')))
    bc_atrib.rename(columns={0:'MA2', 1:'MA3', 2:'MA4', 3:'MA5', 4:'MA6', 5:'MA7', 6:'MA8', 7:'MA9', 8:'MA10', 9:'MA11',
                            10:'MD1', 11:'MD2', 12:'MD3', 13:'MD4', 14:'MD5', 15: 'MD6', 16:'MD7', 17:'MD8', 18:'MD9', 19:'MD10',
                            20:'MD11', 21:'MD12', 22:'MD13', 23:'MD14', 24:'MD15', 25:'MD16', 26:'MD17', 27:'MD18', 28:'MD19'}, inplace = True)
                            #29:'IDMAP', 30:'IPCA', 31:'LAMP', 32:'LMDS', 33:'MDS', 34:'PBC', 35:'TSNE', 36:'UMAP'}, inplace = True)
    #print(bc_atrib)
    
    #print(bc_ranking)

    teste = pd.merge(bc_atrib,bc_ranking, how='inner', left_index=True, right_index=True)
    #print(teste)
    return teste


def knn_mds(df_base_con, k, novo_dataset):
    #print(df_base_con)
    
    #df_bc = base_conhecimento_atributos(df_base_con)
    
    vetor = base_conhecimento_atributos(df_base_con).values #df_bc.values #base_conhecimento_atributos(df_base_con).values #df_base_con.values
    #vetor = df_bc.values 
    
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(vetor)

    #d = np.random.uniform(0.,1.,size=(1,29))
    #print(novo_dataset)

    #print(neigh.kneighbors(novo_dataset))
    neigh_dist, neigh_ind = neigh.kneighbors(novo_dataset)
    
    #print(neigh_ind.ravel())
    #print(df_base_con.iloc[neigh_ind.ravel()])

    df_rk = pd.DataFrame()
    for id in np.sort(neigh_ind.ravel()):
        #print(df_base_con.iloc[id, [29,30,31,32,33,34,35,36]])
        l = df_base_con.iloc[id, [29,30,31,32,33,34,35,36]]  #.astype(int)
        df_rk =  pd.concat([df_rk,l],axis=1)#,ignore_index=True) #rdf.append(l,ignore_index=True)
    #teste = pd.DataFrame(rdf.T)
    #print(teste.dtypes)
    #print(df_rk.T)
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
    #df_base_con_meta_atrib = bc.drop(columns=['BD','MA1','IDMAP', 'IPCA', 'LAMP', 'LMDS', 'MDS', 'PBC', 'TSNE', 'UMAP']) #columns=[0,1,31,32,33,34,35,36,37,38])
    df_base_con_meta_atrib = bc.drop(columns=['IDMAP', 'IPCA', 'LAMP', 'LMDS', 'MDS', 'PBC', 'TSNE', 'UMAP']) #columns=[0,1,31,32,33,34,35,36,37,38])
    #df_base_con_meta_atrib.reset_index(inplace=True, drop=True)
    # ajustando colunas de instancia e dimensão
    
    #df_base_con_meta_atrib['MA2'] = pd.to_numeric(df_base_con_meta_atrib['MA2'])
    #df_base_con_meta_atrib['MA3'] = pd.to_numeric(df_base_con_meta_atrib['MA3'])
    
    #print(df_base_meta)

    # colocando as medidas de nr de instâncias e de dimensões na mesma escala para melhorar a acurácia do Knn
    #colunas = ['MA2','MA3']
    #scaler = MinMaxScaler()
    #df_base_con_meta_atrib[colunas] = scaler.fit_transform(df_base_con_meta_atrib[colunas].astype('float32'))
    #print(df_base_con_meta_atrib)
    return df_base_con_meta_atrib

def validacao_cruzada(bc, k_v, k_f, n_exe):
    bc_norm = normalizar_base_conhecimento(bc)
    #print(bc)
    #print(bc_norm)
    
    indices = list(range(bc_norm.shape[0])) #list(range(df_base_con.shape[0]))
    #print(indices)
    
    # Número de pastas para o k-fold 
    k_folds = k_f

    #kfolds = kfoldcv(indices, k_folds)

    # Número de vezes que a validação cruzada será executada
    n_exe_k_folds = n_exe
    n_kfolds = []
    for i in n_exe_k_folds:
        n_kfolds.append(kfoldcv(indices, k_folds))
        #print(i, 'rodada.')

    scores = []
    scores_pasta = []
    scores_exe = []
    
    exe = 0
    #print(n_kfolds)
    for kfold in n_kfolds:
        exe += 1
        pasta = 0
        print('     Execução ',exe, '   ...')#,'=',np.mean(scores, axis= 0))
        for kf in kfold:
            pasta += 1
            #print('tupla 1')
            #print(kf[0])
            treino_index = []
            for i in kf[0]:
                treino_index.extend(i)
            treino = bc_norm.iloc[treino_index]
            #treino = base_conhecimento_atributos(treino)
            #print(treino)
            #print(kf[1])
            teste = bc_norm.iloc[kf[1]]
            #print(teste)
            teste_atrib = base_conhecimento_atributos(teste)
            #print(teste_atrib)
            


            for i,j in teste_atrib.iterrows():
            
                # transforma a serie unidimensional num array bidimensional para o knn
                j_df = pd.DataFrame(j).T # j.values.reshape(29,1)
                #print(j_df.values)
            
                ranking_predito = knn_mds(treino, k_v, j_df.values)
                #print(pd.DataFrame(ranking_predito).T)
                #print(teste)
                ranking_real = teste.loc[i,'IDMAP':'UMAP'] #,['IDMAP', 'IPCA', 'LAMP', 'LMDS', 'MDS', 'PBC', 'TSNE', 'UMAP']].astype(float)
                #teste.iloc[i,[31,32,33,34,35,36,37,38]].astype(float)
                #print(pd.DataFrame(ranking_real).T)
                score = stats.spearmanr(ranking_predito.squeeze(), ranking_real.squeeze())[0] #pd.DataFrame(ds_t_ranking.iloc[0].values).T.squeeze())
                #print(score)
                scores.append(score)

            print('         Média da pasta', pasta,'=',np.mean(scores, axis= 0))
            scores_pasta.append(np.mean(scores, axis= 0))
            scores = []
        print('     Média da execução', exe,'=',np.mean(scores_pasta, axis= 0))
        scores_exe.append(np.mean(scores_pasta, axis= 0))
        scores_pasta = []
    
    return np.mean(scores_exe, axis= 0)

print('Inicio do processamento')
# carregando a base de conhecimento
df_base_con = pd.read_csv('/home/leo/projetos/knn/base_conhecimento.csv',sep=',')#, header=None)
#print(df_base_con)

ranking_padrao = rk.ranking_padrao(base_conhecimento_ranking(df_base_con))
#print(ranking_padrao)

# Novas recomendações
#ds_teste = pd.read_csv('/home/leo/projetos/knn/base_conhecimento_retirados _para_teste.csv',sep=',')#, header=None)

t_inicio = perf_counter()

#k = 3

# Número de vizinhos do K-NN
k_vizinhos = [i for i in range (1,16)]
print('Parametros k do K-NN:',k_vizinhos)

# Número de pastas para o k-fold 
k_f = 50 #[5, 10, 25, 50]
print('Número de pastas do k-fold:', k_f)

# Número de execuções do k-fold
n_e = 30

''' Número de execuções da validação cruzada'''
n_exe = [i for i in range (1,(n_e + 1))]
print('Número de execuções do k-fold:', n_exe)

scores_por_k = [] 
for k_v in k_vizinhos:
    print('k = ', k_v)
    scores_por_k.append(validacao_cruzada(df_base_con, k_v, k_f, n_exe))
    print('k = ', k_v, ' score = ', scores_por_k[k_v-1])
    print('_________________________________________________________')

#print(scores_por_k)
print('Coeficiente padrão: ',np.mean(scores_por_k, axis= 0))
#media = np.full(k_vizinhos.count,np.mean(scores_por_k, axis= 0))

# exportando o resultado para geração do grafico
acuracia = np.array([k_vizinhos, scores_por_k], dtype=object)
#print(acuracia)
acuracia = pd.DataFrame(acuracia).T
#print(acuracia)
acuracia[0] = acuracia[0].astype(int)
acuracia.to_csv('%s/acuracia_final_%sx_%sf_%sk.csv' %(os.getcwd(), len(n_exe), k_f, len(k_vizinhos)),index=None, header=None)


'''
# Validação cruzada k-fold (5 pastas)
indices = list(range(df_base_con.shape[0]))
#print(indices)
k_folds = 5

kfolds = kfoldcv(indices, k_folds)
scores = []
#print(kfolds)

for kf in kfolds:
    #print('tupla 1')
    #print(kf[0])
    treino_index = []
    for i in kf[0]:
        treino_index.extend(i)
    treino = df_base_con.iloc[treino_index]
    #treino = base_conhecimento_atributos(treino)
    #print(treino)
    #print(kf[1])
    teste = df_base_con.iloc[kf[1]]
    #print(teste)
    teste_atrib = base_conhecimento_atributos(teste)
    #print(teste_atrib)

    for i,j in teste_atrib.iterrows():
       
        # transforma a serie unidimensional num array bidimensional para o knn
        j_df = pd.DataFrame(j).T # j.values.reshape(29,1)
        #print(j_df.values)
       
        ranking_predito = knn_mds(treino, k, j_df.values)
        #print(pd.DataFrame(ranking_predito).T)
        #print(teste)
        ranking_real = teste.loc[i,'IDMAP':'UMAP'] #,['IDMAP', 'IPCA', 'LAMP', 'LMDS', 'MDS', 'PBC', 'TSNE', 'UMAP']].astype(float)
        #teste.iloc[i,[31,32,33,34,35,36,37,38]].astype(float)
        #print(pd.DataFrame(ranking_real).T)
        score = stats.spearmanr(ranking_predito.squeeze(), ranking_real.squeeze()) #pd.DataFrame(ds_t_ranking.iloc[0].values).T.squeeze())
        #print(score)
        scores.append(score)

#print(scores)
print(np.mean(scores, axis= 0))
'''


t_fim = perf_counter()
print("Tempo de processamento Dataset k-fold: k-,k," ,(t_fim-t_inicio), ' s')
    

'''for v in teste.values:
       #print(v.reshape(1,teste.shape[1]))
       print(v)
       ranking_predito = knn_mds(treino, k, v.reshape(1,base_conhecimento_atributos(teste)))
       #print(teste)teste.shape[1]))
       print(ranking_predito)
       score = stats.spearmanr(ranking_predito.squeeze(), ranking_padrao.squeeze()) #pd.DataFrame(ds_t_ranking.iloc[0].values).T.squeeze())
       scores.append(score)
'''



