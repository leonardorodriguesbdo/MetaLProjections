# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tempfile
import subprocess
import os
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

def extracao_meta_atributos(nome_ds, ds):
    # Envia o dataset para ser processado pelo executável do ferrari e retorna um dataframe com os meta-atributos
    envia_dados(nome_ds, ds)
    if os.path.exists("./results/%s_MA_Distance.data" %(nome_ds)):
        MAD = pd.read_table("./results/%s_MA_Distance.data" %(nome_ds), delim_whitespace=True, header=None)
    else:
        MAD = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #    print('Arquivo dos meta-atributos não encontrado')

    return MAD


def envia_dados(nome_ds, ds):
    #scaler = MinMaxScaler()
    #ds = scaler.fit_transform(ds.astype('float32'))
    df = pd.DataFrame(ds)
    '''for i,j in df.iterrows():
        #print(j)
        if j[0] == 0.260869:
            print(j)
        if df[i].isna().sum() > 0:
            print(j)
    print(df.isna().sum())'''
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_file = open(tmp_dir.name + '/' + nome_ds + '.data', 'w+')
    
    df.to_csv(tmp_file.name, index=None, header=False) 
    
    path_dataset = tmp_file.name  #caminho e nome do banco de dados
    command = './MAferrari/MADistance' #executavel do ferrari
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




# calcula MD_1
def media_w(W):
    return np.mean(W)

# calcula MD_2
def variancia_w(W):
    return np.var(W)

# calcula MD_3
def desvio_padrao_w(W):
    return np.std(W)

# calcula MD_4
def assimetria_w(W):
    return stats.skew(W)

# calcula MD_5
def curtose_w(W):
    return stats.kurtosis(W, fisher=False)

# calcula MD_6
def perc_h1(x):
    return x[0]/sum(x)

# calcula MD_7
def perc_h2(x):
    return x[1]/sum(x)

# calcula MD_8
def perc_h3(x):
    return x[2]/sum(x)

# calcula MD_9
def perc_h4(x):
    return x[3]/sum(x)

# calcula MD_10
def perc_h5(x):
    return x[4]/sum(x)

# calcula MD_11
def perc_h6(x):
    return x[5]/sum(x)

# calcula MD_12
def perc_h7(x):
    return x[6]/sum(x)

# calcula MD_13
def perc_h8(x):
    return x[7]/sum(x)

# calcula MD_14
def perc_h9(x):
    return x[8]/sum(x)

# calcula MD_15
def perc_h10(x):
    return x[9]/sum(x)



