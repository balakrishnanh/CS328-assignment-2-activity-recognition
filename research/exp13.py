import sys; sys.path.insert(0,'research')
from evalkit import *
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix
L=5
C=pd.read_pickle(f'research/cache3_Class_Data_{L}.pkl').replace([np.inf,-np.inf],0).fillna(0)
allf=[c for c in C.columns if c not in('activity','recording')]
labs=['walk','upstairs','downstairs','jogging','standing','sitting','lying']
P=np.zeros((len(C),7))
for a,b in GroupKFold(5).split(C,groups=C.recording):
    m=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,random_state=0).fit(C.iloc[a][allf],C.iloc[a].activity)
    P[b]=m.predict_proba(C.iloc[b][allf]); cls=m.classes_
pred=cls[P.argmax(1)]
print('win',(pred==C.activity).mean())
print(pd.DataFrame(confusion_matrix(C.activity,pred,labels=labs),index=labs,columns=labs))
# smoothing over neighbors within recording
for k in [1,2,3]:
    Ps=pd.DataFrame(P).groupby(C.recording.values).transform(lambda s:s.rolling(2*k+1,center=True,min_periods=1).mean()).values
    print('smooth',k,(cls[Ps.argmax(1)]==C.activity).mean())
Pr=pd.DataFrame(P).groupby(C.recording.values).transform('mean').values
print('full-recording mean proba (per window)',(cls[Pr.argmax(1)]==C.activity).mean())
r=pd.DataFrame(P,columns=cls).groupby(C.recording.values).mean();t=C.groupby('recording').activity.first()
print('recording acc (mean proba)',(r.idxmax(1)==t.loc[r.index]).mean())
