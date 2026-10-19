import sys; sys.path.insert(0,'research')
from evalkit import *
L=5
S,C=ld('Sample_Data',L),ld('Class_Data',L)
allf=[c for c in C.columns if c not in('activity','recording','t_start','rec_dur','fs')]
def inner(df,k): return df[(df.t_start>=k)&(df.t_start+L<=df.rec_dur-k)]
ev=inner(C,5)  # evaluation always on interior windows
for k in [0,2,5,8]:
    tr=inner(C,k) if k else C
    accs=[]
    for a,b in GroupKFold(5).split(C,groups=C.recording):
        A=C.iloc[a];B=C.iloc[b]; A=inner(A,k) if k else A; B=inner(B,5)
        m=mk('et').fit(A[allf],A.activity);accs.append((m.predict(B[allf])==B.activity.values).mean())
    m=mk('et').fit(tr[allf],tr.activity);Si=inner(S,5)
    print('trim train',k,'gcv interior',round(np.mean(accs),3),'C->S interior',round((m.predict(Si[allf])==Si.activity.values).mean(),3),len(tr),flush=True)
# per-recording weighting
w=1/C.groupby('recording').recording.transform('size')
accs=[]
for a,b in GroupKFold(5).split(C,groups=C.recording):
    m=mk('et');m.fit(C.iloc[a][allf],C.iloc[a].activity,sample_weight=w.iloc[a]);accs.append((m.predict(C.iloc[b][allf])==C.iloc[b].activity.values).mean())
print('recweighted gcv',round(np.mean(accs),3))
