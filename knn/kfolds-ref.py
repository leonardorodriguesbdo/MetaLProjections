import random

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
        #print('fold: ', i, ' ', train, test)
        #print('__________________________________________________________________________________')
        
    return kfolds

indices = list(range(10))
k = 2

n_vezes = [i for i in range (1,10)]
for i in n_vezes:
    print(kfoldcv(indices, k))
    #print('')