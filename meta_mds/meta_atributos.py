# -*- coding: utf-8 -*-i
import os
import csv
#import time
#import timeit
import numpy as np
import pandas as pd
import scipy as sci
from scipy import stats, spatial
import calcula_ma as cma
import calcula_md as cmd
from metrics import *
import vp
import math
from sklearn import datasets
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid
from sys import stdout
import projections
import joblib
import operator
from time import perf_counter
from collections import deque
import sys 

def lista_projections():
    return sorted(projections.all_projections.keys())
    
def run_eval(dataset_name, projection_name, X, y, output_dir):
    # TODO: add noise adder
    global DISTANCES

    dc_results = dict()
    pq_results = dict()
    projected_data = dict()
    lst_mu = []

    dc_results['original'] = eval_dc_metrics(
       X=X, y=y, dataset_name=dataset_name, output_dir=output_dir)
    
    proj_tuple = projections.all_projections[projection_name]
    proj = proj_tuple[0]
    grid_params = proj_tuple[1]
    
    grid = ParameterGrid(grid_params)
    
    for params in grid:
        id_run = proj.__class__.__name__ + '-' + str(params)
        proj.set_params(**params)

        print('-----------------------------------------------------------------------')
        print(projection_name, id_run)

        X_new, y_new, result = projections.run_projection(
              proj, X, y, id_run, dataset_name, output_dir)

        
        pq_results[id_run] = result
        projected_data[id_run] = dict()
        projected_data[id_run]['X'] = X_new
        projected_data[id_run]['y'] = y_new
    
    
    #results_to_dataframe(dc_results, dataset_name).to_csv(
    #    '%s/%s_%s_dc_results.csv' % (output_dir, dataset_name, projection_name), index=None)
    #results_to_dataframe(pq_results, dataset_name).to_csv(
    #    '%s/%s_%s_pq_results.csv' % (output_dir, dataset_name, projection_name), index=None)
    df_results, max_mu = results_to_dataframe_calcula_mu(pq_results, dataset_name) 
    df_results.to_csv('%s/%s_%s_pq_results.csv' % (output_dir, dataset_name, projection_name), index=None)
   
    # Não tem utilização
    #joblib.dump(projected_data, '%s/%s_%s_projected.pkl' %
    #            (output_dir, dataset_name, projection_name))
    #joblib.dump(DISTANCES, '%s/%s_%s_distance_files.pkl' %
    #            (output_dir, dataset_name, projection_name))

    #remove_files_proc()
    
    return max_mu
            
def mqs_to_dataframe(mqs, dataset_name, t):
    df = pd.DataFrame([mqs])
    column_list = df.columns
    df['dataset_name'] = dataset_name
    df['tempo'] = t
    df = df.reset_index(drop=True)
    df = df.loc[:, ['dataset_name'] + list(column_list) + ['tempo']]
    return df

def carrega_dataset(dataset_name):
        data_dir = os.path.join('data', dataset_name)

        #Pegar o meta-atributo MA1
        arq = open(os.path.join(data_dir, 'meta.txt'), 'r')
        for linha in arq:
            if linha[:6] == 'Tipo: ':
                tipo = linha[6:]

        #print([l for l in arq if l[:6] == 'Tipo: '])        
        X = np.load(os.path.join(data_dir, 'X.npy'), allow_pickle=True)
        #XC = np.load(os.path.join(data_dir, 'XC.npy'))
        y = np.load(os.path.join(data_dir, 'y.npy'), allow_pickle=True)

        return X, y, tipo.replace('\n', ''), dataset_name

#Normaliza os valores de uma coluna (Será usado na coluna stress e shepard)
def normalize_col(col):
    scaler = MinMaxScaler()
    X = np.array(np.array(col).reshape((-1, 1)))
    X[np.isnan(X)] = -1
    X = scaler.fit_transform(X)
    return X.squeeze()

#Substitui o metodo results_to_dataframe do arquivo metrics.py 
def results_to_dataframe_calcula_mu(results, dataset_name):
    df = pd.DataFrame.from_dict(results).transpose()
    column_list = df.columns

    df['proj'] = df.index + '-'
    df['projection_name'] = df['proj'].apply(lambda x: pd.Series(str(x).split('-')))[0]
    df['projection_parameters'] = df['proj'].apply(
        lambda x: pd.Series(str(x).split('-')))[1]
    df['dataset_name'] = dataset_name
    df = df.drop(['proj'], axis=1)

    if df['projection_name'].values[0] == 'IncrementalPCA':
        if df['metric_pq_shepard_diagram_correlation'].values[0] < 0. :
            df['metric_pq_shepard_diagram_correlation'] = 0.
        if df['metric_pq_shepard_diagram_correlation'].values[0] > 1. :
            df['metric_pq_shepard_diagram_correlation'] = 1.

        if df['metric_pq_normalized_stress'].values[0] < 0. :
            df['metric_pq_normalized_stress'] = 0.
        if df['metric_pq_normalized_stress'].values[0] > 1. :
            df['metric_pq_normalized_stress'] = 1.



    if df['metric_pq_shepard_diagram_correlation'].min() < 0. or df['metric_pq_shepard_diagram_correlation'].max() > 1:
        df['metric_pq_shepard_diagram_correlation'] = normalize_col(df['metric_pq_shepard_diagram_correlation'])
    if df['metric_pq_normalized_stress'].min() < 0 or df['metric_pq_normalized_stress'].max() > 1:
        df['metric_pq_normalized_stress'] = normalize_col(df['metric_pq_normalized_stress'])
    
    #Calcula o valor de MU
    df['mu'] = (df['metric_pq_continuity_k_07'] + df['metric_pq_trustworthiness_k_07'] + df['metric_pq_neighborhood_hit_k_07'] +
                df['metric_pq_shepard_diagram_correlation'] + (1 - df['metric_pq_normalized_stress']))/5

    df = df.reset_index(drop=True)
    df = df.loc[:, ['dataset_name', 'projection_name',
                    'projection_parameters'] + list(column_list) + ['mu']]
    
    return df, max(df['mu'])

# Normalização do dataset. Escala [0,1]
def dataset_Normalizado(Ds):
    df_min_max_scaled = Ds.copy() 
    for column in df_min_max_scaled.columns: 
        df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())     
    return df_min_max_scaled

# Vetor d da similaridade entre as instancias
def vetor_dist_normalizado(d):
    df_min_max_scaled = d.copy() 
    df_min_max_scaled = (df_min_max_scaled - df_min_max_scaled.min()) / (df_min_max_scaled.max() - df_min_max_scaled.min())     
    return df_min_max_scaled

#def meta_atributos(nome, tipo, Ds, y, wl, projection_name):
def meta_atributos(nome, tipo, X, y, projection_name):
        #print(Ds)
        arredonda = 5
        
        #Nome base
        ma = d
        
        print('----------- calculando MA ----------------------')
        # MA_1 tipo de dado
        ma1 = tipo #cma.tipo_dado(t)
        
        # MA_2 numero de linhas
        #ma2 = cma.total_linhas(X) 
        ma2 = round(cma.total_linhas(X), arredonda) #round(math.log(cma.total_linhas(Ds), 2), arredonda)

        # MA_3 numero de dimensões
        #ma3 = cma.total_dimensoes(X, y)
        ma3 = round(cma.total_dimensoes(X), arredonda)  #X, y), 2), arredonda) #round(math.log(cma.total_dimensoes(Ds), 2), arredonda)  #X, y), 2), arredonda)

        # MA_4 taxa dimensionalidade intrinseca
        ma4 = round(cma.taxa_dim_intrinseca(X), arredonda)
        
        # MA_5 razão de dispersão
        ma5 = round(cma.razao_dispersao(X), arredonda)

        #print(Ds)

        MA = cma.extracao_meta_atributos(nome, X) #pd.DataFrame(np.zeros((1, 10))) 
        if not MA.empty:
            # MA_6 porcentagem de outliers
            ma6 = round(MA.iloc[0,4], arredonda)

            # MA_7 média da entropia dos atributos discretos
            ma7 = round(MA.iloc[0,5], arredonda)

            # MA_8 média da concentração entre os atributos discretos
            ma8 = round(MA.iloc[0,6], arredonda)

            # MA_9 correlação media absoluta entre atributos continuos
            ma9 = round(MA.iloc[0,7], arredonda)

            # MA_10 Assimetria média de atributos continuos
            ma10 = round(MA.iloc[0,8], arredonda)
               
            # MA_11 Curtose media dos atributos continuos
            ma11 = round(MA.iloc[0,9], arredonda)
        
        '''
        #print(MA)
        #print(Ds)
        
        MD = cmd.extracao_meta_atributos(nome, Ds)
        
        # MD_1 Media de w'
        md1 = round(MD.iloc[0,1], arredonda)
        # MD_2 Variancia de w'
        md2 = round(MD.iloc[0,2], arredonda)
        # MD_3 Desvio padrão de w'
        md3 = round(MD.iloc[0,3], arredonda)
        # MD_4 Assimetria de w'
        md4 = round(MD.iloc[0,4], arredonda)
        # MD_5 Curtose de w'
        md5 = round(MD.iloc[0,5], arredonda)
        # MD_6 percentual intervalo [0,0.1]'
        md6 = round(MD.iloc[0,6], arredonda)
        # MD_7 percentual intervalo (0.1,0.2]'
        md7 = round(MD.iloc[0,7], arredonda)
        # MD_8 percentual intervalo (0.2,0.3]'
        md8 = round(MD.iloc[0,8], arredonda)
        # MD_9 percentual intervalo (0.3,0.4]'
        md9 = round(MD.iloc[0,9], arredonda)
        # MD_10 percentual intervalo (0.4,0.5]'
        md10 = round(MD.iloc[0,10], arredonda)
        # MD_11 percentual intervalo (0.5,0.6]'
        md11 = round(MD.iloc[0,11], arredonda)
        # MD_12 percentual intervalo (0.6,0.7]'
        md12 = round(MD.iloc[0,12], arredonda)
        # MD_13 percentual intervalo (0.7,0.8]'
        md13 = round(MD.iloc[0,13], arredonda)
        # MD_14 percentual intervalo (0.8,0.9]'
        md14 = round(MD.iloc[0,14], arredonda)
        # MD_15 percentual intervalo (0.9,1]'
        md15 = round(MD.iloc[0,15], arredonda)
        # MD_16 percentual escore-z absoluto intervalo [0,1)'
        md16 = round(MD.iloc[0,16], arredonda)
        # MD_17 percentual escore-z absoluto intervalo [1,2)'
        md17 = round(MD.iloc[0,17], arredonda)
        # MD_18 percentual escore-z absoluto intervalo [2,3)'
        md18 = round(MD.iloc[0,18], arredonda)
        # MD_19 percentual escore-z absoluto intervalo [3,--)'
        md19 = round(MD.iloc[0,19], arredonda)

        #print(MD)
        
        
        #print(Ds)
        '''
        
        #t1_inicio = perf_counter()
        print('----------- calculando MD ----------------------')
        t2_inicio = perf_counter()
        # Normaliza a base caso necessário
        if not(X.min() >= 0.0 and X.max() <= 1.0):
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X.astype('float32'))
        
        w = spatial.distance.pdist(X, metric='euclidean')
        '''
        D = [] 
        cor = deque()
        N = cma.total_linhas(X)
        l1 = 0
        l2 = 1
        i = 0
        tam = len(w)
        for i in range(tam):
            cor.append(stats.spearmanr(X[l1], X[l2])[0])
            l2+=1
            if l2 == N:
                l1+=1
                l2 = l1 + 1
        '''    
        # vetor de distancia e vetor de correlaçao     
        #w = np.concatenate((w, cor))
        # retira nan do array
        #w =  w[~np.isnan(w)]
        
        wl = vetor_dist_normalizado(w)
        
        # MD_1 Media de w'
        md1 = round(cmd.media_w(wl), arredonda)
        # MD_2 Variancia de w'
        md2 = round(cmd.variancia_w(wl), arredonda)
        # MD_3 Desvio padrão de w'
        md3 = round(cmd.desvio_padrao_w(wl), arredonda)
        # MD_4 Assimetria de w'
        md4 = round(cmd.assimetria_w(wl), arredonda)
        # MD_5 Curtose de w'
        md5 = round(cmd.curtose_w(wl), arredonda)
        
        d_hist = np.histogram(wl)#,bins=10,range=(0.0,1.0))
        x = d_hist[0]

        # MD_6 percentual intervalo [0,0.1]'
        md6 = round(cmd.perc_h1(x), arredonda)
        # MD_7 percentual intervalo (0.1,0.2]'
        md7 = round(cmd.perc_h2(x), arredonda)
        # MD_8 percentual intervalo (0.2,0.3]'
        md8 = round(cmd.perc_h3(x), arredonda)
        # MD_9 percentual intervalo (0.3,0.4]'
        md9 = round(cmd.perc_h4(x), arredonda)
        # MD_10 percentual intervalo (0.4,0.5]'
        md10 = round(cmd.perc_h5(x), arredonda)
        # MD_11 percentual intervalo (0.5,0.6]'
        md11 = round(cmd.perc_h6(x), arredonda)
        # MD_12 percentual intervalo (0.6,0.7]'
        md12 = round(cmd.perc_h7(x), arredonda)
        # MD_13 percentual intervalo (0.7,0.8]'
        md13 = round(cmd.perc_h8(x), arredonda)
        # MD_14 percentual intervalo (0.8,0.9]'
        md14 = round(cmd.perc_h9(x), arredonda)
        # MD_15 percentual intervalo (0.9,1]'
        md15 = round(cmd.perc_h10(x), arredonda)

        #Z-Score = (d[0]-md1) / md3
        z = stats.zscore(wl)
        y1 = sum(0 <= x < 1 for x in np.absolute(z))
        y2 = sum(1 <= x < 2 for x in np.absolute(z))
        y3 = sum(2 <= x < 3 for x in np.absolute(z))
        y4 = sum(3 <= x for x in np.absolute(z))
        yt = y1 + y2 + y3 + y4

        # MD_16 percentual escore-z absoluto intervalo [0,1)'
        md16 = round((y1/yt), arredonda)
        # MD_17 percentual escore-z absoluto intervalo [1,2)'
        md17 = round((y2/yt), arredonda)
        # MD_18 percentual escore-z absoluto intervalo [2,3)'
        md18 = round((y3/yt), arredonda)
        # MD_19 percentual escore-z absoluto intervalo [3,--)'
        md19 = round((y4/yt), arredonda)
        
        t2_fim = perf_counter()
        time = round(t2_fim-t2_inicio)

        # Gera o arquivo dataset_MD_Distance.data
        temp = [] 
        temp.append([time, md1, md2, md3, md4, md5, md6, md7, md8, md9, md10, md11, md12, md13, md14, md15, md16, md17, md18, md19])

        file = "./results/%s_MA_Distance.data" %(d)
        temp = pd.DataFrame(temp)
        temp.to_csv(file, sep ='\t', index = False, header = False, encoding = 'utf-8')
        
        
        # Extrai métricas de qualidade para cada projeção no dataset
        print('----------- calculando MQ ----------------------')
        t2_inicio = perf_counter()
        
        # Normaliza o dataset
        #scaler = MinMaxScaler()
        #X = scaler.fit_transform(Ds.astype('float32'))
                                
        mqs = dict()
        for p in projection_name:
            mqs[p] = run_eval(d, p, X, y, output_dir_met_ds_proj)
        
        #guarda os valores de mu para cada dataset e projeção no arquivo csv
        t2_fim = perf_counter()
        mqs_to_dataframe(mqs, d, (t2_fim-t2_inicio) ).to_csv(
                '%s/%s_mq_results.csv' % (output_dir_res_met_qual, d), index=None)
        
        #mqs = [0,0,0,0,0,0,0,0]
        # organiza as métricas na base de conhecimento de acordo as 8 projeções escolhidas e monta o ranking
        mq1, mq2, mq3, mq4, mq5, mq6, mq7, mq8 = 0, 0, 0, 0, 0, 0, 0, 0

        j = 1 #inicia o ranking  
        sort_mqs = sorted(mqs, key = mqs.get, reverse=True) 
        for i in sort_mqs:
            #print(i, mqs[i])
            if i == "IDMAP":
                mq1= j
            elif i == "IPCA":
                mq2= j
            elif i == "LAMP":
                mq3= j
            elif i == "LMDS":
                mq4= j
            elif i == "MDS":
                mq5= j
            elif i == "PBC":
                mq6= j
            elif i == "TSNE":
                mq7= j
            elif i == "UMAP":
                mq8= j
            else:
                print("Não foi possível ranquear ", i)
            j += 1
        
        dados.append([ma,ma1, ma2, ma3, ma4, ma5, ma6, ma7, ma8, ma9, ma10, ma11,
                         md1, md2, md3, md4, md5, md6, md7, md8, md9, md10, md11, md12, md13, md14, md15, md16, md17, md18, md19,
                         mq1, mq2, mq3, mq4, mq5, mq6, mq7, mq8])
        print('----------------------------------------------------------------------------')
        

if __name__ == '__main__':
    # inicio
    path_ds = os.path.join(os.getcwd(), 'data')
    projection_name = lista_projections()
    output_dir_met_ds_proj = os.path.join('metricas_ds_proj')
    output_dir_res_met_qual = os.path.join('results')
    datasets = np.sort(os.listdir(path_ds))
    dados = []
    #datasets = ['svhn_ok','spambase_ok','fmd_ok','sentiment_ok','coil20_ok','imdb_ok','arrhythmia_ok']
    #datasets = ['bank','cnae9','acute','water_treatement'] #, 'spatial', 'libra8','acute','water_treatement'] 
    #datasets = ['libra8', 'acute', 'agua', 'zoo'] 

    # Abre o arquivo da base de conhecimento guarda o conteúdo ja processado
    with open('base_conhecimento.csv', encoding='utf-8') as base_conhecimento:
        tabela = csv.reader(base_conhecimento, delimiter=',')
        for l in tabela:
            dados.append(l)
    
    print("Processando meta-atributos...")
    t_inicio = perf_counter()

    #datasets_ferrari = np.sort(os.listdir(os.path.join(os.getcwd(),'results')))
    
    for d in datasets: #['communities_ok']: 
        # Verifica se o dataset ja foi processado, caso sim pula para o próximo
        if 'ok' in d:
            continue

        t1_inicio = perf_counter()
          
        X, y, tipo, nome = carrega_dataset(d)
        #print(X)
        #print(y)
               
        stdout.write(nome)
        stdout.write(' .....')
        print('.')
        
        #meta_atributos(nome, tipo, X, y, wl, projection_name)
        meta_atributos(nome, tipo, X, y, projection_name)
        
        #renomear a pasta sinalizando seu processamento
        nome_antigo = os.path.join(path_ds, nome) #path + '\\' + nome
        nome_novo = os.path.join(path_ds, nome + '_ok') #path + '\\' + nome + '_ok'
        os.rename(nome_antigo, nome_novo)

        # Atualiza o arquivo csv com os dados processados
        f = open('base_conhecimento.csv', 'w', newline='', encoding='utf-8')
        w = csv.writer(f)
        for l in dados:
            w.writerow(l)
        f.close()
        
        t1_fim = perf_counter()
        print("Tempo de processamento Dataset: ", d, ': ',(t1_fim-t1_inicio), ' s')
        

    print('--------------------------------------------------------------------')
    print("Processamento Finalizado.")
    t_fim = perf_counter()
    print("Tempo de processamento: ",(t_fim-t_inicio), ' s')
    #print(df)
    print('--------------------------------------------------------------------')
    
