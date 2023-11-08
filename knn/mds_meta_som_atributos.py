import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Minisom library and module is used for performing Self Organizing Maps
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

def atributos_MA(data):
    # Defining X variables for the input of SOM
    X = data.iloc[:, 2:12].values
    # X variables:
    print(pd.DataFrame(X))
    sc = MinMaxScaler(feature_range = (0, 1))
    X = sc.fit_transform(X)
    print(X)
    return X

def atributos_MA_individual(X):
    
    for (columnName, columnData) in X.iteritems():
        if columnName not in ['BD','MA1']:            
            print('Column Name : ', columnName)
            print('Column Contents : ', columnData)
            col = pd.DataFrame(columnData)
            sc = MinMaxScaler(feature_range = (0, 1))
            X = sc.fit_transform(col)
            graf_som_atributos(X,columnName)
    '''for c in X:
        print(c)
        graf_som_atributos(c,'atr')'''

def atributos_MD(data):
    # Defining X variables for the input of SOM
    X = data.iloc[:, 12:32].values
    # X variables:
    print(pd.DataFrame(X))
    sc = MinMaxScaler(feature_range = (0, 1))
    X = sc.fit_transform(X)
    print(X)
    return X 

def atributos_combinado(data):
    # Defining X variables for the input of SOM
    X = data.iloc[:, 2:31].values
    # X variables:
    print(pd.DataFrame(X))
    sc = MinMaxScaler(feature_range = (0, 1))
    X = sc.fit_transform(X)
    print(X)
    return X  

def atributos_MQ(data):
    print(data)
    # Defining X variables for the input of SOM
    X = data.iloc[:, 1:9].values
    # X variables:
    print(pd.DataFrame(X))
    #sc = MinMaxScaler(feature_range = (0, 1))
    #X = sc.fit_transform(X)
    print(X)
    return X 

def graf_som_atributos(X, atributo):
    print(X)
    # Initialization and training
    n_neurons = 10
    m_neurons = 10

    som = MiniSom(n_neurons, m_neurons, X.shape[1], sigma=1.5, learning_rate=.5, 
                neighborhood_function='gaussian', random_seed=0)

    som.pca_weights_init(X)
    som.train(X, 1000, verbose=True)  # random training


    plt.figure(figsize=(7, 7))
    frequencies = som.activation_response(X)
    plt.pcolor(frequencies.T, cmap='Blues') 
    plt.colorbar()

    #plt.title("Frequência de problemas em cada neurônio do mapa - " + 'Atributo %s' %(atributo))
    #plt.legend(loc='best', shadow=True)
    
    plt.savefig('resulting_images/som_atributo_%s.png' %(atributo))
    plt.show()

# inicio
data = pd.read_csv('/home/leo/projetos/knn/base_conhecimento.csv',sep=',')
print(data)


X = atributos_MA(data)
graf_som_atributos(X,'MA')

X = atributos_MD(data)
graf_som_atributos(X,'MD')

X = atributos_combinado(data)
graf_som_atributos(X,'Combinado')

dataMQ = pd.read_csv('/home/leo/projetos/knn/MQ_base_conhecimento.csv',sep=',')
print(dataMQ)

X = atributos_MQ(dataMQ)
graf_som_atributos(X,'MQ')

#atributos_MA_individual(data)
#graf_som_atributos(X,'Ind')

print()




