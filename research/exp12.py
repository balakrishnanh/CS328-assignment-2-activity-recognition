import sys; sys.path.insert(0,'research')
from evalkit import *
from sklearn.ensemble import HistGradientBoostingClassifier
import evalkit
_mk=evalkit.mk
def mk2(n):
    if n=='hgb': return HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,random_state=0)
    if n=='et1000': return ExtraTreesClassifier(600,random_state=0,n_jobs=-1,max_features=0.3)
    return _mk(n)
evalkit.mk=mk2
for L in [5,10]:
    C=pd.read_pickle(f'research/cache3_Class_Data_{L}.pkl').replace([np.inf,-np.inf],0).fillna(0)
    S=pd.read_pickle(f'research/cache3_Sample_Data_{L}.pkl').replace([np.inf,-np.inf],0).fillna(0)
    allf=[c for c in C.columns if c not in('activity','recording')]
    for nm in ['et','et1000','hgb','rf']:
        print(L,nm,len(allf),'split %.3f'%split(C,allf,nm),'gcv',[round(float(v),3) for v in gcv(C,allf,nm)],'C->S',[round(float(v),3) for v in xdom(C,S,allf,nm)],flush=True)
