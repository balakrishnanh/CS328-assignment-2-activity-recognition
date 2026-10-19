import sys; sys.path.insert(0,'research')
from evalkit import *
from sklearn.ensemble import ExtraTreesClassifier
C=pd.read_pickle('research/cache3_Class_Data_5.pkl').replace([np.inf,-np.inf],0).fillna(0)
allf=[c for c in C.columns if c not in('activity','recording')]
folds=list(GroupKFold(5).split(C,groups=C.recording))
def score(fs):
    acc=[]
    for a,b in folds:
        m=ExtraTreesClassifier(120,random_state=0,n_jobs=-1,min_samples_leaf=2).fit(C.iloc[a][fs],C.iloc[a].activity);acc.append(m.score(C.iloc[b][fs],C.iloc[b].activity))
    return np.mean(acc)
sel=[]
for step in range(20):
    r={f:score(sel+[f]) for f in allf if f not in sel}
    f=max(r,key=r.get);sel.append(f);print(step,f,round(r[f],4),flush=True)
print(sel)
