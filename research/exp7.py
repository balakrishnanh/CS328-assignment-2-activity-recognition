import sys; sys.path.insert(0,'research')
from evalkit import *
from sklearn.ensemble import ExtraTreesClassifier
L=5
S,C=ld('Sample_Data',L),ld('Class_Data',L)
allf=[c for c in C.columns if c not in('activity','recording','t_start','rec_dur','fs')]
folds=list(GroupKFold(5).split(C,groups=C.recording))
def score(fs):
    acc=[]
    for a,b in folds:
        m=ExtraTreesClassifier(100,random_state=0,n_jobs=-1,min_samples_leaf=2).fit(C.iloc[a][fs],C.iloc[a].activity);acc.append(m.score(C.iloc[b][fs],C.iloc[b].activity))
    return np.mean(acc)
sel=[];best=0
for step in range(14):
    r={f:score(sel+[f]) for f in allf if f not in sel}
    f=max(r,key=r.get);sel.append(f);print(step,f,round(r[f],4),flush=True)
