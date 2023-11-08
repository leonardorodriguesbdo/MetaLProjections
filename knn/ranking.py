import pandas as pd
import numpy as np
import os
import csv

'''
def checagem_empate(bc):
    path_results = '/home/leo/projetos/meta_mds/results/'
    datasets = np.sort(os.listdir(path_results))

    for d in datasets:
        if d.find('_results.csv') == -1:
            continue #print('não processa')
        else:
            #print('processa')
            df = pd.read_csv(os.path.join(path_results,d), sep=',')
            dataset = df.values[0][0]
            #print(dataset)
            #print(df)
            #df_1 = df.drop(0)
            df_1 = df.drop(columns=['dataset_name','tempo'])
            #print(df_1)
            rdf = df_1.rank(axis=1,method='average',ascending=False)
            #print(rdf)

            #print(bc)
            #print(bc.query("BD == 'ads'"))
            id = bc.index[bc['BD'] == dataset].tolist()
            bc.loc[id, ['IDMAP_1']] = rdf['IDMAP'][0]
            bc.loc[id, ['IPCA_1']] = rdf['IPCA'][0]
            bc.loc[id, ['LAMP_1']] = rdf['LAMP'][0]
            bc.loc[id, ['LMDS_1']] = rdf['LMDS'][0]
            bc.loc[id, ['MDS_1']] = rdf['MDS'][0]
            bc.loc[id, ['PBC_1']] = rdf['PBC'][0]
            bc.loc[id, ['TSNE_1']] = rdf['TSNE'][0]
            bc.loc[id, ['UMAP_1']] = rdf['UMAP'][0]

        #print(bc)

    bc.to_csv('teste.csv', index=None)
'''
def organiza_ranking_individual(rk_ind):
    #print(rk_ind)
    global log_erro_valor
    log_erro_valor = 'sem erro na funçao'
    if rk_ind['metric_pq_neighborhood_hit_k_07'] < 0 or rk_ind['metric_pq_neighborhood_hit_k_07'] > 1:
        log_erro_valor = rk_ind['dataset_name'] + '\n'
    if rk_ind['metric_pq_trustworthiness_k_07'] < 0 or rk_ind['metric_pq_trustworthiness_k_07'] > 1:
        log_erro_valor = rk_ind['dataset_name'] + '\n'
    if rk_ind['metric_pq_continuity_k_07'] < 0 or rk_ind['metric_pq_continuity_k_07'] > 1:
        log_erro_valor = rk_ind['dataset_name'] + '\n'
    if rk_ind['metric_pq_normalized_stress'] < 0 or rk_ind['metric_pq_normalized_stress'] > 1:
        log_erro_valor = rk_ind['dataset_name'] + '\n'
    if rk_ind['metric_pq_shepard_diagram_correlation'] < 0 or rk_ind['metric_pq_shepard_diagram_correlation'] > 1:
        log_erro_valor = rk_ind['dataset_name'] + '\n'
    lst = [rk_ind['metric_pq_neighborhood_hit_k_07'],
           rk_ind['metric_pq_trustworthiness_k_07'],
           rk_ind['metric_pq_continuity_k_07'],
           rk_ind['metric_pq_normalized_stress'],
           rk_ind['metric_pq_shepard_diagram_correlation']] 
    return pd.DataFrame(lst,index=['mq1','mq2','mq3','mq4','mq5'], columns=[rk_ind['projection_name']])

def results_media_MQ_individual(path_results_ind):
    arquivos = np.sort(os.listdir(path_results_ind))
    tmp = []    
    #ds = ''
    global tot_indice_ind
    for a in arquivos:
    #print(a)
        if a.find('_mq_individual.csv') != -1: 
            ds = a[:a.find('_mq')] 
            print('Gerando indices individuais:',a)
            caminho = os.path.join(path_results_ind,a)
            df = pd.read_csv(caminho,sep=',')
            temp = df.mean()
            
            # Convert a serie em uma lista
            lst = temp.tolist()
            # Inserindo na posição 1
            lst.insert(0,ds)
            # Converte a lista para serie
            temp = pd.Series(lst)
            
            tmp.append(temp)
            
    df = pd.DataFrame(tmp)
    df.rename(columns={0: 'BD'}, inplace = True)
    df.rename(columns={1: 'IDMAP'}, inplace = True)
    df.rename(columns={2: 'IPCA'}, inplace = True)
    df.rename(columns={3: 'LAMP'}, inplace = True)
    df.rename(columns={4: 'LMDS'}, inplace = True)
    df.rename(columns={5: 'MDS'}, inplace = True)
    df.rename(columns={6: 'PBC'}, inplace = True)
    df.rename(columns={7: 'TSNE'}, inplace = True)
    df.rename(columns={8: 'UMAP'}, inplace = True)
    print(df)
    df.to_csv('%s/MQ_base_conhecimento.csv' %(os.path.join(path)),index=None)

def results_MQ_individual(path_results_ind):
    arquivos = np.sort(os.listdir(path_results_ind))
    tmp = pd.DataFrame() 
    ds = ''
    global tot_indice_ind
    for a in arquivos:
        #print(a)
        if a.find('_pq_results.csv') != -1:
            print('Gerando indices individuais:',a)
            if a.find('_IDMAP') != -1:
                ds = a[:a.find('_IDMAP')]
                caminho = os.path.join(path_results_ind,a)
                df = pd.read_csv(caminho,sep=',')
                idmap = organiza_ranking_individual(df.iloc[df[['mu']].idxmax()].squeeze())
                print(idmap)                
            elif a.find('_IPCA') != -1:
                caminho = os.path.join(path_results_ind,a)
                df = pd.read_csv(caminho,sep=',')
                ipca = organiza_ranking_individual(df.iloc[df[['mu']].idxmax()].squeeze())
                #ipca = organiza_ranking_individual(df.max())
                #print(ipca)
            elif a.find('_LAMP') != -1:
                caminho = os.path.join(path_results_ind,a)
                df = pd.read_csv(caminho,sep=',')
                lamp = organiza_ranking_individual(df.iloc[df[['mu']].idxmax()].squeeze())
                #print(lamp)
            elif a.find('_LMDS') != -1:
                caminho = os.path.join(path_results_ind,a)
                df = pd.read_csv(caminho,sep=',')
                lmds = organiza_ranking_individual(df.iloc[df[['mu']].idxmax()].squeeze())
                #print(lmds)
            elif a.find('_MDS') != -1:
                caminho = os.path.join(path_results_ind,a)
                df = pd.read_csv(caminho,sep=',')
                mds = organiza_ranking_individual(df.iloc[df[['mu']].idxmax()].squeeze())
                #print(mds)
            elif a.find('_PBC') != -1:
                caminho = os.path.join(path_results_ind,a)
                df = pd.read_csv(caminho,sep=',')
                pbc = organiza_ranking_individual(df.iloc[df[['mu']].idxmax()].squeeze())
                #print(pbc)
            elif a.find('_TSNE') != -1:
                caminho = os.path.join(path_results_ind,a)
                df = pd.read_csv(caminho,sep=',')
                tsne = organiza_ranking_individual(df.iloc[df[['mu']].idxmax()].squeeze())
                #print(tsne)
            elif a.find('_UMAP') != -1:
                caminho = os.path.join(path_results_ind,a)
                df = pd.read_csv(caminho,sep=',')
                umap = organiza_ranking_individual(df.iloc[df[['mu']].idxmax()].squeeze())
                tmp = pd.concat([idmap,ipca,lamp,lmds,mds,pbc,tsne,umap],axis=1, join='inner')
                #print(tmp)
                tmp.to_csv('%s/%s_mq_individual.csv' %(os.path.join(path,'results MQ individual'),ds))
                tot_indice_ind += 1

def results_rank_individual(path_rk_ind):
    arquivos = np.sort(os.listdir(path_rk_ind))
    global tot_rank_ind
    for a in arquivos:
        if a.find('_mq_individual.csv') != -1:
            print('Gerando Ranking individual:',a)
            ds = a[:a.find('_mq')]
            caminho = os.path.join(path_rk_ind,a)
            df = pd.read_csv(caminho,sep=',')
            df = df.drop(columns=['Unnamed: 0'])
            # ajusta a linha de mq3 pois a escala ranquea pelo mais próximo de zero
            df.iloc[3] = df.iloc[3]*-1
            rdf = df.rank(axis=1,method='average',ascending=False)
            rdf.insert(0, "MQ", ['mq1','mq2','mq3','mq4','mq5'], True)
            #print(rdf)
            rdf.to_csv('%s/%s_rank_individual.csv' %(os.path.join(path_knn,'result MQ individual'),ds),index=None)
            tot_rank_ind += 1

def ranking_medio_por_metrica(path_rk_ind):
    arquivos = np.sort(os.listdir(path_rk_ind))
    print('Gerando Ranking médio individual por métrica:')
    global tot_rank_ind
    df_m1, df_m2, df_m3, df_m4, df_m5 = [],[],[],[],[]
    for a in arquivos:
        if a.find('_rank_individual.csv') != -1:
            #print('Gerando Ranking médio individual por métrica:',a)
            ds = a[:a.find('_rank')]
            caminho = os.path.join(path_rk_ind,a)
            df = pd.read_csv(caminho,sep=',')
            #print(df)
            df_m1.append(df.iloc[0].values)# = pd.concat([df_m1,df.loc[0]], axis = 1, join='inner')
            df_m2.append(df.iloc[1].values)# = pd.concat([df_m1,df.loc[0]], axis = 1, join='inner')
            df_m3.append(df.iloc[2].values)# = pd.concat([df_m1,df.loc[0]], axis = 1, join='inner')
            df_m4.append(df.iloc[3].values)# = pd.concat([df_m1,df.loc[0]], axis = 1, join='inner')
            df_m5.append(df.iloc[4].values)# = pd.concat([df_m1,df.loc[0]], axis = 1, join='inner')
            #rdf.to_csv('%s/%s_rank_individual.csv' %(os.path.join(path_knn,'result MQ individual'),ds),index=None)
            #tot_rank_ind += 1
    df_m1 = pd.DataFrame(df_m1)
    df_m1.to_csv('%s/%s_rank_individual.csv' %(path,'mq1'),index=None)
    df_m1 = df_m1.drop(columns=[0])
    rm_df_m1 = ranking_medio(df_m1)
    rm_df_m1.to_csv('%s/%s_rank_medio.csv' %(path,'mq1'),index=None)

    df_m2 = pd.DataFrame(df_m2)
    df_m2.to_csv('%s/%s_rank_individual.csv' %(path,'mq2'),index=None)
    df_m2 = df_m2.drop(columns=[0])
    rm_df_m2 = ranking_medio(df_m2)
    rm_df_m2.to_csv('%s/%s_rank_medio.csv' %(path,'mq2'),index=None)
    
    df_m3 = pd.DataFrame(df_m3)
    df_m3.to_csv('%s/%s_rank_individual.csv' %(path,'mq3'),index=None)
    df_m3 = df_m3.drop(columns=[0])
    rm_df_m3 = ranking_medio(df_m3)
    rm_df_m3.to_csv('%s/%s_rank_medio.csv' %(path,'mq3'),index=None)

    df_m4 = pd.DataFrame(df_m4)
    df_m4.to_csv('%s/%s_rank_individual.csv' %(path,'mq4'),index=None)
    df_m4 = df_m4.drop(columns=[0])
    rm_df_m4 = ranking_medio(df_m4)
    rm_df_m4.to_csv('%s/%s_rank_medio.csv' %(path,'mq4'),index=None)

    df_m5 = pd.DataFrame(df_m5)
    df_m5.to_csv('%s/%s_rank_individual.csv' %(path,'mq5'),index=None)
    df_m5 = df_m5.drop(columns=[0])
    rm_df_m5 = ranking_medio(df_m5)
    rm_df_m5.to_csv('%s/%s_rank_medio.csv' %(path,'mq5'),index=None)

    tmp = pd.concat([rm_df_m1,rm_df_m2,rm_df_m3,rm_df_m4,rm_df_m5],axis=0, join='inner')
    tmp.rename(columns={1: 'IDMAP' , 2: 'IPCA', 3: 'LAMP', 4:'LMDS',
                        5: 'MDS', 6: 'PBC', 7: 'TSNE', 8: 'UMAP' }, inplace = True)
    tmp.insert(0, "MQ", ['mq1','mq2','mq3','mq4','mq5'], True)
    print(tmp)
    tmp.to_csv('%s/%s_rank_medio.csv' %(path,'mqs'),index=None)

    #print(pd.DataFrame(df_m2))
    #print(pd.DataFrame(df_m3))
    #print(pd.DataFrame(df_m4))
    #print(pd.DataFrame(df_m5))
        

def ranking_final_meta_exemplos(path_rk_ind):
    arquivos = np.sort(os.listdir(path_rk_ind))
    bc = pd.read_csv(os.path.join(path,'base_conhecimento.csv'),sep=',')
    #print(bc)
    #global tot_rank_ind
    rank_bc = pd.DataFrame()
    for a in arquivos:
        if a.find('_rank_individual.csv') != -1:
            print('Gerando Ranking Médio individual:',a)
            ds = a[:a.find('_rank_')]
            caminho = os.path.join(path_rk_ind,a)
            df = pd.read_csv(caminho,sep=',')
            #print(df)
            df = df.drop(columns=['MQ'])
            #print('ranking individual')
            #print(df)
            rdf = pd.DataFrame(df.mean()).T
            #print('Calculo da media dos rankings individuais')
            #print(rdf)
            rdf = rdf.rank(axis=1,method='average',ascending=True)
            #print('ranking medio')
            #print(rdf)
            rdf.insert(0, "dataset_name", [ds], True)
            #print(rdf)

            id = bc.index[bc['BD'] == ds].tolist()
            #print(id)
            bc.loc[id, ['IDMAP_1']] = rdf['IDMAP'][0]
            bc.loc[id, ['IPCA_1']] = rdf['IncrementalPCA'][0]
            bc.loc[id, ['LAMP_1']] = rdf['LAMP'][0]
            bc.loc[id, ['LMDS_1']] = rdf['LandmarkMDS'][0]
            bc.loc[id, ['MDS_1']] = rdf['MDS'][0]
            bc.loc[id, ['PBC_1']] = rdf['ProjectionByClustering'][0]
            bc.loc[id, ['TSNE_1']] = rdf['MTSNE'][0]
            bc.loc[id, ['UMAP_1']] = rdf['UMAP'][0]
        else:
            continue
        #rank_bc = pd.concat([rdf],axis=0, join='inner')
        rank_bc = pd.concat([rank_bc,rdf],ignore_index=True)
    #bc = pd.concat(bc,rank_bc,ignore_index=True)
    print(rank_bc)
    print(bc)
    
    # Gera arquivo csv do ranking das projeções para cada dataset
    rank_bc.to_csv('%s/ranking_base_conhecimento.csv' %(path),index=None)
    bc.to_csv('%s/base_conhecimento_final.csv' %(path),index=None)

def ranking_medio(df_rank):
    #print(df_rank)
    #print(df_rank.mean())
    temp = pd.DataFrame(df_rank.mean()).T
    #print(temp)
    # o menor valor de média ocupa a primeira posição no ranking
    temp = temp.rank(method='average',ascending=True, axis=1)
    #print(temp)
    return temp

def ranking_predito(df_rank):
    return ranking_medio(df_rank).T

def ranking_padrao(df_rank):
    return ranking_medio(df_rank)

def ranking_meta_exemplos(bc):
    rank = bc[['IDMAP', 'IPCA', 'LAMP', 'LMDS', 'MDS', 'PBC', 'TSNE', 'UMAP']]
    return rank


if __name__ == '__main__':
    # inicio
    print('Iniciando procesamento de ranking...')
    # path principal
    path = os.getcwd()
    # path para montagem do rankin individual
    path_results_ind = '/home/leo/projetos/meta_mds/metricas_ds_proj/'
    path_rk_ind = os.path.join(path,'results MQ individual') 

    tot_indice_ind = 0
    tot_rank_ind = 0

    log_erro_valor = 'nenhum erro'
    #results_MQ_individual(path_results_ind)
    #results_rank_individual(path_rk_ind)
    #ranking_medio_por_metrica(path_rk_ind)
    #ranking_final_meta_exemplos(path_rk_ind)
    #results_media_MQ_individual(path_rk_ind)
    #print(log_erro_valor)
    #print(tot_indice_ind, ' arquivos de indice individual gerado(s)')
    #print(tot_rank_ind, ' arquivos de ranking individual gerado(s)')

    
    '''
    b_c = pd.read_csv(os.path.join(path_knn,'base_conhecimento.csv'),sep=',')
    print(b_c)
    rk_padrao = ranking_padrao(ranking_meta_exemplos(b_c))
    print(rk_padrao)
    #print(ranking_padrao(rank))
    '''

