# Referêrncia: https://medium.com/data-hackers/como-criar-k-fold-cross-validation-na-m%C3%A3o-em-python-c0bb06074b6b
import numpy as np
import pandas as pd
import ranking as rk
import mds_meta_frequencia_relativa as freq_rel
import os
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from time import perf_counter
from collections import Counter

# retorna a base de conhecimento com a parte dos rankings
def base_conhecimento_ranking(bc):
    return bc[['IDMAP', 'IPCA', 'LAMP', 'LMDS', 'MDS', 'PBC', 'TSNE', 'UMAP']].astype(float) #df_base_con[[31,32,33,34,35,36,37,38]]

def graf_curva_validacao_multiplas_linhas(acuracia, baseline_spearman):
    #print(acuracia)
        
    plt.title("Curva de validação com classificador KNN")
    plt.xlabel("Número de vizinhos")
    plt.ylabel("Acurácia")
    plt.tight_layout()

    plt.grid()    
    sns.lineplot(data=acuracia, x=0, y=1, hue=2 )

    plt.legend(loc='best', shadow=True, title='')
    plt.xlim(0, 16)
    plt.xticks(range(1, 16, 1)) # alterar escala do eixo
    #plt.ylim(0.75, 0.9)
    #plt.yticks(np.arange(0.7, 0.9, 0.05))
    plt.savefig('resulting_images/acuracia_multiplas_linhas.png')
    plt.show()
    
    

def graf_curva_validacao_knn(arquivo, acuracia, baseline_spearman):
    #print(acuracia)
    #plt.close()

    y = acuracia[1].values
    x = acuracia[0].values

    t = np.full(y.size,baseline_spearman[0])

    plt.plot(x, y, 'bs-', x, t, 'r--', )
        
    # Creating the plot
    #plt.title("Curva de validação com classificador KNN " + arquivo)
    plt.xlabel("Número de vizinhos")
    plt.ylabel("Acurácia")
    plt.tight_layout()
    plt.legend(('k-NN', 'SRC padrão'), loc='best', shadow=True)
    plt.xlim(1, 16)
    plt.xticks(range(1, 16, 1)) # alterar escala do eixo
    plt.ylim(0.75, 0.9)
    plt.yticks(np.arange(0.7, 0.9, 0.05))
    plt.savefig('resulting_images/acuracia_%s.png' %(arquivo))
    plt.show()
    
def graf_atributo_dimensao(bc):
    print(bc)
    # MA2 - nr de linhas  MA3 - nr de dimensões
    bc = bc[['MA2', 'MA3']]
    bc.insert(2, 'Dimensionalidade', np.nan)
    condicoes = [(bc['MA3'] < 100), 
             ((bc['MA3'] >= 100) & (bc['MA3'] <= 500)), 
             (bc['MA3'] > 500)]
    opcoes = ['baixa', 'media', 'alta']
    bc['Dimensionalidade'] = np.select(condicoes, opcoes)
    bc['MA2'] = np.log10(bc['MA2'])
    bc['MA3'] = np.log10(bc['MA3'])
    print(bc)

    #list(range(0,4,0.25))
    
    sns.relplot(data=bc, x="MA2", y="MA3", hue='Dimensionalidade', palette=["b", "g", "r"])
    #plt.legend(['Baixa','Média', 'Alta'], title = "")
    plt.xticks(np.arange(5),[1,10,100,1000,10000])
    plt.xlabel('log10')
    plt.yticks(np.arange(5),[1,10,100,1000,10000])
    plt.xlabel( "Nr de instancias \n base log10" , size = 12 ) 
    plt.ylabel( "Nr de dimensões \n base log10" , size = 12 )
    plt.savefig('resulting_images/grafico_nr_dimensoes.png')
    
                           
def baseline(rank_bc): # rank_padrao):
    #print(rank_bc)
    rk_padrao = rk.ranking_padrao(rank_bc)
    #print(rk_padrao)
    scores = []
    for i, rk_real in rank_bc.iterrows():
        #print(rk_real)#rank_bc.loc[i,'IDMAP':'UMAP'])
        score = stats.spearmanr(rk_real, rk_padrao.squeeze()) #pd.DataFrame(ds_t_ranking.iloc[0].values).T.squeeze())
        #print(score)
        scores.append(score)
    #print(np.mean(scores, axis= 0))
    return np.mean(scores, axis= 0)

    

print('Inicio do processamento')

#grafico atributoxdimensao
bc = pd.read_csv('/home/leo/projetos/knn/base_conhecimento.csv',sep=',')
graf_atributo_dimensao(bc)
print('fim')

# carregando a base de conhecimento
'''df_mq1 = pd.read_csv('/home/leo/projetos/knn/mq1_rank_individual.csv',sep=',')
df_mq1 = df_mq1.drop(columns=['0'])
df_mq2 = pd.read_csv('/home/leo/projetos/knn/mq2_rank_individual.csv',sep=',')
df_mq2 = df_mq2.drop(columns=['0'])
df_mq3 = pd.read_csv('/home/leo/projetos/knn/mq3_rank_individual.csv',sep=',')
df_mq3 = df_mq3.drop(columns=['0'])
df_mq4 = pd.read_csv('/home/leo/projetos/knn/mq4_rank_individual.csv',sep=',')
df_mq4 = df_mq4.drop(columns=['0'])
df_mq5 = pd.read_csv('/home/leo/projetos/knn/mq5_rank_individual.csv',sep=',')
df_mq5 = df_mq5.drop(columns=['0'])

#df_rank_bc = rk.ranking_meta_exemplos(pd.read_csv('/home/leo/projetos/knn/base_conhecimento.csv',sep=','))
df_rank_bc = rk.ranking_meta_exemplos(bc)
df_rank_bc.to_csv('/home/leo/projetos/knn/erromedia.csv')
print(df_rank_bc.mean())
baseline_spearman = baseline(df_rank_bc)

#acuracia = pd.read_csv('/home/leo/projetos/knn/acuracia_final.csv', sep=',', header=None)
arquivos = np.sort(os.listdir(os.getcwd()))
for a in arquivos:
    if a.find('acuracia_final_') != -1:
        print('Gerando grafico do arquivo:',a)
        # carregando a base de acuracia
        acuracia = pd.read_csv('/home/leo/projetos/knn/%s'  %(a), sep=',', header=None)
        graf_curva_validacao_knn(a, acuracia=acuracia, baseline_spearman=baseline_spearman)
        
acuracia = pd.read_csv('/home/leo/projetos/knn/acuracia_final.csv', sep=',', header=None)
graf_curva_validacao_multiplas_linhas(acuracia=acuracia, baseline_spearman=baseline_spearman)
'''
'''
df_rank_bc.rename(columns={'IDMAP': '1'}, inplace = True)
df_rank_bc.rename(columns={'IPCA': '2'}, inplace = True)
df_rank_bc.rename(columns={'LAMP': '3'}, inplace = True)
df_rank_bc.rename(columns={'LMDS': '4'}, inplace = True)
df_rank_bc.rename(columns={'MDS': '5'}, inplace = True)
df_rank_bc.rename(columns={'PBC': '6'}, inplace = True)
df_rank_bc.rename(columns={'TSNE': '7'}, inplace = True)
df_rank_bc.rename(columns={'UMAP': '8'}, inplace = True)
#print(df_rank_bc)



freq_rel.graf_freq_relativa(df_mq1, 'mq1', 100)
freq_rel.graf_freq_relativa(df_mq2, 'mq2', 100)
freq_rel.graf_freq_relativa(df_mq3, 'mq3', 100)
freq_rel.graf_freq_relativa(df_mq4, 'mq4', 100)
freq_rel.graf_freq_relativa(df_mq5, 'mq5', 100)
freq_rel.graf_freq_relativa(df_rank_bc, 'rm', 100)
'''
print('fim do processamento')