import sys; sys.path.insert(0,'research')
from evalkit import *
import evalkit
evalkit.mk0=evalkit.mk
for L in [1,2,3,5,10]:
    S,C=ld('Sample_Data',L),ld('Class_Data',L)
    allf=[c for c in C.columns if c not in('activity','recording')]
    out=[L]
    for name in ['dt5','et']:
        out+= [name,'split',round(split(C,allf,name),3),'gcv',[round(float(v),3) for v in gcv(C,allf,name)],'C->S',[round(float(v),3) for v in xdom(C,S,allf,name)],'S->C',[round(float(v),3) for v in xdom(S,C,allf,name)]]
    print(*out,flush=True)
