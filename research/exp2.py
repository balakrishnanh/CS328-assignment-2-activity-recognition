import sys; sys.path.insert(0,'research')
from evalkit import *
from sklearn.metrics import classification_report, confusion_matrix
L=5
S,C=ld('Sample_Data',L),ld('Class_Data',L)
allf=[c for c in C.columns if c not in('activity','recording')]
print(len(allf), allf)
print(C.activity.value_counts().to_dict()); print(S.activity.value_counts().to_dict())
for name in ['dt5','rf','et','knn','lr','svm','gb']:
    for fs_name,fs in [('base',BASE),('all',allf)]:
        print(name,fs_name,'split',round(split(C,fs,name),3),'gcv',[round(float(v),3) for v in gcv(C,fs,name)],'C->S',[round(float(v),3) for v in xdom(C,S,fs,name)],'S->C',[round(float(v),3) for v in xdom(S,C,fs,name)],flush=True)
