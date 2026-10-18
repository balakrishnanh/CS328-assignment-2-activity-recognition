import sys; sys.path.insert(0,'research')
import pandas as pd, numpy as np
C=pd.read_pickle('research/cache3_Class_Data_5.pkl')
D=C[C.activity.isin(['lying','sitting','standing'])]
r=D.groupby('recording').agg(a=('activity','first'),lf=('lf_tot','median'),tot=('tot_std','median'))
r['lf_q']=pd.cut(r.lf,[0,0.003,0.01,0.03,0.1,10])
print(pd.crosstab(r.lf_q,r.a))
print(r.groupby('a').lf.describe().round(4))
