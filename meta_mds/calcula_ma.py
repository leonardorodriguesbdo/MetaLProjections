# -*- coding: utf-8 -*-
import subprocess
import os
import tempfile
import pandas as pd
import numpy as np
from pandas import DataFrame
#from scipy.stats import entropy
from sklearn.decomposition import PCA

# calcula MA_1
def tipo_dado(t):
    return t

def total_objetos(Ds):
    return Ds.size

# calcula MA_2 (samples)
def total_linhas(Ds):
    return Ds.shape[0]

# calcula MA_3 (Features)
def total_dimensoes(Ds):
    # junta X com y formando um unico dataframe
    #Ds = pd.DataFrame(X)
    #Ds[X.shape[1]] = y

    return Ds.shape[1]

def metric_dc_num_classes(y):
    return len(np.unique(y))

# calcula MA_4
def taxa_dim_intrinseca(Ds):
    pca = PCA()
    pca.fit(Ds)

    #return np.where(pca.explained_variance_ratio_.cumsum() >= 0.95)[0][0] + 1
    return (np.where(pca.explained_variance_ratio_.cumsum() >= 0.95)[0][0] + 1) / Ds.shape[1]    

# calcula MA_5
def razao_dispersao(Ds):
    return 1.0 - (np.count_nonzero(Ds) / float(Ds.shape[0] * Ds.shape[1]))

def extracao_meta_atributos(nome_ds, ds):
    # Envia o dataset para ser processado pelo executável do ferrari e retorna um dataframe com os meta-atributos
    envia_dados(nome_ds, ds)
    if os.path.exists("./results/%s_MA_Attributes.data" %(nome_ds)):
        MAA = pd.read_table("./results/%s_MA_Attributes.data" %(nome_ds), delim_whitespace=True, header=None)
    else:
        MAA = pd.DataFrame(np.zeros((1, 10)))
    #    print('Arquivo dos meta-atributos não encontrado')

    return MAA


def envia_dados(nome_ds, ds):
    df = DataFrame(ds)
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_file = open(tmp_dir.name + '/' + nome_ds + '.data', 'w+')
    
    df.to_csv(tmp_file.name, index=None, header=False) 
    
    path_dataset = tmp_file.name  #caminho e nome do banco de dados
    command = './MAferrari/MAAttributes' #executavel do ferrari
    cmdline = [command, path_dataset]

    rc = subprocess.run(cmdline, universal_newlines=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=86400, check=True)
    
    if rc.returncode != 0: #except subprocess.CalledProcessError as rc:
        print('return code: ', rc.returncode)
        print('stdout:')
        print('_________________________________________________')
        print(rc.stdout)
        print('_________________________________________________')
        print('stderr:')
        print('_________________________________________________')
        print(rc.stderr)
        print('#################################################')


    

