import numpy as np
import pandas as pd
import scipy.stats as stats

cre_DF = pd.read_csv('/Users/yazanb/Dropbox/Allen Institute/Manuscripts/DRNs/drn_exp.csv')

print cre_DF
AM = np.array(cre_DF.percent)[cre_DF.area == 'VISam']
AL = np.array(cre_DF.percent)[cre_DF.area == 'VISal']
P = np.array(cre_DF.percent)[cre_DF.area == 'VISp']
L = np.array(cre_DF.percent)[cre_DF.area == 'VISl']
PM = np.array(cre_DF.percent)[cre_DF.area == 'VISpm']
RL = np.array(cre_DF.percent)[cre_DF.area == 'VISrl']

print len(AM) + len(AL) + len(P) + len(L) + len(PM) + len(RL)
print stats.f_oneway(AM, AL, P, L, PM, RL)


Cux2 = np.array(cre_DF.percent)[cre_DF.cre == 'Cux2-CreERT2']
Emx1 = np.array(cre_DF.percent)[cre_DF.cre == 'Emx1-IRES-Cre']
Fezf2 = np.array(cre_DF.percent)[cre_DF.cre == 'Fezf2-CreER']
Nr5a1 = np.array(cre_DF.percent)[cre_DF.cre == 'Nr5a1-Cre']
Ntsr1 = np.array(cre_DF.percent)[cre_DF.cre == 'Ntsr1-Cre_GN220']
Rbp4 = np.array(cre_DF.percent)[cre_DF.cre == 'Rbp4-Cre_KL100']
Rorb = np.array(cre_DF.percent)[cre_DF.cre == 'Rorb-IRES2-Cre']
Scnn1a = np.array(cre_DF.percent)[cre_DF.cre == 'Scnn1a-Tg3-Cre']
Slc = np.array(cre_DF.percent)[cre_DF.cre == 'Slc17a7-IRES2-Cre']
Sst = np.array(cre_DF.percent)[cre_DF.cre == 'Sst-IRES-Cre']
Tlx = np.array(cre_DF.percent)[cre_DF.cre == 'Tlx3-Cre_PL56']
Vip = np.array(cre_DF.percent)[cre_DF.cre == 'Vip-IRES-Cre']

print stats.f_oneway(Cux2, Emx1, Fezf2, Nr5a1, Ntsr1, Rbp4, Rorb, Scnn1a, Slc, Sst, Tlx, Vip)
