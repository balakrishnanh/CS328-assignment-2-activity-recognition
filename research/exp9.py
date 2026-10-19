import sys; sys.path.insert(0,'research')
from evalkit import *
L=5
C=ld('Class_Data',L)
allf=[c for c in C.columns if c not in('activity','recording','t_start','rec_dur','fs')]
# 1) data quality: recordings whose overall intensity contradicts the label
r=C.groupby('recording').agg(a=('activity','first'),tot=('tot_std','median'),n=('tot_std','size'),dur=('rec_dur','first'))
static=r.a.isin(['lying','sitting','standing'])
print('static recs with gait-level motion (median tot_std>1.5):',(static&(r.tot>1.5)).sum(),'/',static.sum())
print('gait recs with static-level motion (median tot_std<0.5):',(~static&(r.tot<0.5)).sum(),'/',(~static).sum())
print(r[~static&(r.tot<0.8)].sort_values('tot').head(15))
print(r[static&(r.tot>1.0)].sort_values('tot').tail(15))
# 2) context features: window features + recording medians
med=C.groupby('recording')[allf].transform('median').add_prefix('rec_')
q90=C.groupby('recording')[allf].transform(lambda s:s.quantile(0.9)).add_prefix('r90_')
CC=pd.concat([C,med,q90],axis=1)
for name,fs in [('win',allf),('win+recmed',allf+list(med.columns)),('win+recmed+q90',allf+list(med.columns)+list(q90.columns)),('recmed only',list(med.columns))]:
    print(name,[round(float(v),3) for v in gcv(CC,fs,'et')])
