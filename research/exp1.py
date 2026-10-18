import sys; sys.path.insert(0,'research')
from evalkit import *
for L in [1,2,3,5,10]:
    S,C=ld('Sample_Data',L),ld('Class_Data',L)
    print(L,'n',len(S),len(C),'split',round(split(C,BASE,'dt5'),3),'gcv',[round(v,3) for v in gcv(C,BASE,'dt5')],'C->S',[round(v,3) for v in xdom(C,S,BASE,'dt5')],'S->C',[round(v,3) for v in xdom(S,C,BASE,'dt5')])
