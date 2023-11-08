import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def graf_freq_relativa(bc, metrica, lim_eixo):
    # ajustes no processamento da técnica IDMAP (preenchimento do valor zero para rank vazio)    
    bc['1'] = bc['1'].apply(np.floor)
    idmap = bc['1'].value_counts(normalize=True).sort_index() * 100
    if metrica in ['mq4']:
        idmap[7.0] = 0
        idmap[8.0] = 0
    idmap = idmap.sort_index()
    df_idmap = pd.DataFrame(idmap)
    df_idmap.rename(columns={'1': 'qtde'}, inplace = True)
    df_idmap['rank'] = np.full(8, ['1 °','2 °','3 °','4 °','5 °','6 °','7 °','8 °'])
    df_idmap['tecnica'] = np.full(8,'IDMAP')
    
    # ajustes no processamento da técnica IPCA (preenchimento do valor zero para rank vazio)
    bc['2'] = bc['2'].apply(np.floor)
    ipca = bc['2'].value_counts(normalize=True).sort_index() * 100
    if metrica in ['mq1','mq2','mq4']:
        ipca[1.0] = 0        
    if metrica in ['mq4']:
        ipca[2.0] = 0
        ipca[4.0] = 0
    ipca = ipca.sort_index()
    df_ipca = pd.DataFrame(ipca)
    df_ipca.rename(columns={'2': 'qtde'}, inplace = True)
    df_ipca['rank'] = np.full(8, ['1 °','2 °','3 °','4 °','5 °','6 °','7 °','8 °'])#np.full(8, np.arange(1,9))
    df_ipca['tecnica'] = np.full(8,'IPCA')
    
    # ajustes no processamento da técnica LAMP (preenchimento do valor zero para rank vazio)
    bc['3'] = bc['3'].apply(np.floor)
    lamp = bc['3'].value_counts(normalize=True).sort_index() * 100
    if metrica in ['mq2','mq4']:
        lamp[1.0] = 0
    if metrica in ['mq4']:
        lamp[2.0] = 0
    lamp = lamp.sort_index()
    df_lamp = pd.DataFrame(lamp)
    df_lamp.rename(columns={'3': 'qtde'}, inplace = True)
    df_lamp['rank'] = np.full(8, ['1 °','2 °','3 °','4 °','5 °','6 °','7 °','8 °'])#np.full(8, np.arange(1,9))
    df_lamp['tecnica'] = np.full(8,'LAMP')
    
    # ajustes no processamento da técnica LMDS (preenchimento do valor zero para rank vazio)
    bc['4'] = bc['4'].apply(np.floor)
    lmds = bc['4'].value_counts(normalize=True).sort_index() * 100
    if metrica in ['mq1','mq2','mq4','rm']:
        lmds[1.0] = 0
    if metrica in ['mq4']:
        lmds[2.0] = 0
        lmds[3.0] = 0
        lmds[5.0] = 0
    lmds = lmds.sort_index()
    df_lmds = pd.DataFrame(lmds)
    df_lmds.rename(columns={'4': 'qtde'}, inplace = True)
    df_lmds['rank'] = np.full(8, ['1 °','2 °','3 °','4 °','5 °','6 °','7 °','8 °'])#np.full(8, np.arange(1,9))
    df_lmds['tecnica'] = np.full(8,'LMDS')

    # ajustes no processamento da técnica MDS (preenchimento do valor zero para rank vazio)
    bc['5'] = bc['5'].apply(np.floor)
    mds = bc['5'].value_counts(normalize=True).sort_index() * 100
    if metrica in ['mq1','mq2','mq3','mq4']:
        mds[1.0] = 0
    if metrica in ['mq4']:
        mds[7.0] = 0
        mds[8.0] = 0
    mds = mds.sort_index()
    df_mds = pd.DataFrame(mds)
    df_mds.rename(columns={'5': 'qtde'}, inplace = True)
    df_mds['rank'] = np.full(8, ['1 °','2 °','3 °','4 °','5 °','6 °','7 °','8 °'])#np.full(8, np.arange(1,9))
    df_mds['tecnica'] = np.full(8,'MDS')

    # ajustes no processamento da técnica PBC (preenchimento do valor zero para rank vazio)
    bc['6'] = bc['6'].apply(np.floor)
    pbc = bc['6'].value_counts(normalize=True).sort_index() * 100
    if metrica in ['mq1','mq3','rm']:
        pbc[1.0] = 0
    if metrica in ['mq5']:
        pbc[2.0] = 0
    pbc = pbc.sort_index()
    df_pbc = pd.DataFrame(pbc)
    df_pbc.rename(columns={'6': 'qtde'}, inplace = True)
    df_pbc['rank'] = np.full(8, ['1 °','2 °','3 °','4 °','5 °','6 °','7 °','8 °'])#np.full(8, np.arange(1,9))
    df_pbc['tecnica'] = np.full(8,'PCB')

    # ajustes no processamento da técnica TSNE (preenchimento do valor zero para rank vazio)
    bc['7'] = bc['7'].apply(np.floor)
    tsne = bc['7'].value_counts(normalize=True).sort_index() * 100
    if metrica in ['mq2','mq4']:
        tsne[8.0] = 0
    if metrica in ['mq4']:
        tsne[4.0] = 0
        tsne[5.0] = 0
        tsne[7.0] = 0
    if metrica in ['rm']:
        tsne[6.0] = 0
        tsne[7.0] = 0
    tsne = tsne.sort_index()
    df_tsne = pd.DataFrame(tsne)
    df_tsne.rename(columns={'7': 'qtde'}, inplace = True)
    df_tsne['rank'] = np.full(8, ['1 °','2 °','3 °','4 °','5 °','6 °','7 °','8 °'])#np.full(8, np.arange(1,9))
    df_tsne['tecnica'] = np.full(8,'TSNE')

    # ajustes no processamento da técnica UMAP (preenchimento do valor zero para rank vazio)
    bc['8'] = bc['8'].apply(np.floor)
    umap = bc['8'].value_counts(normalize=True).sort_index() * 100
    if metrica == 'mq1':
        umap[5.0] = 0
        umap[7.0] = 0
    umap = umap.sort_index()
    df_umap = pd.DataFrame(umap)
    df_umap.rename(columns={'8': 'qtde'}, inplace = True)
    df_umap['rank'] = np.full(8, ['1 °','2 °','3 °','4 °','5 °','6 °','7 °','8 °'])#np.full(8, np.arange(1,9))
    df_umap['tecnica'] = np.full(8,'UMAP')
    
    # Organizando o dataframe para utilização no grafico de barras 
    df = pd.merge(df_idmap, df_ipca, how = 'outer')
    df = pd.merge(df, df_lamp, how = 'outer')
    df = pd.merge(df, df_lmds, how = 'outer')
    df = pd.merge(df, df_mds, how = 'outer')
    df = pd.merge(df, df_pbc, how = 'outer')
    df = pd.merge(df, df_tsne, how = 'outer')
    df = pd.merge(df, df_umap, how = 'outer')

    df.to_csv('%s/%s_df_freq_relativa_mqs.csv' %(os.getcwd(), metrica),index=None, header=None)
    
    #sns.set_context('paper')
    #sns.set_color_codes('dark')
    graf = sns.barplot(x ='tecnica', y = 'qtde', data = df, 
                hue = "rank" )
    plt.legend(title = "Ranking", loc = 2, bbox_to_anchor = (1,1), frameon = False)
    plt.xlabel(metrica) 
    plt.ylabel('%') 
    plt.ylim(0, lim_eixo)
    graf.figure.set_size_inches(10, 6)
    plt.savefig('resulting_images/freq_relativa_%s.png' %(metrica))
    plt.show()
