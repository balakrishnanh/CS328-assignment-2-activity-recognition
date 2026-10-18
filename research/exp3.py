import sys; sys.path.insert(0,'research')
from evalkit import *
from sklearn.metrics import classification_report, confusion_matrix
L=5
S,C=ld('Sample_Data',L),ld('Class_Data',L)
allf=[c for c in C.columns if c not in('activity','recording')]
labs=['walk','upstairs','downstairs','jogging','standing','sitting','lying']
P=np.empty(len(C),dtype=object)
for a,b in GroupKFold(5).split(C,groups=C.recording):
    m=mk('rf').fit(C.iloc[a][allf],C.iloc[a].activity);P[b]=m.predict(C.iloc[b][allf])
print(pd.DataFrame(confusion_matrix(C.activity,P,labels=labs),index=labs,columns=labs))
print(classification_report(C.activity,P))
m=mk('rf').fit(C[allf],C.activity);p=m.predict(S[allf]);print('C->S');print(pd.DataFrame(confusion_matrix(S.activity,p,labels=labs),index=labs,columns=labs))
print(pd.Series(m.feature_importances_,allf).sort_values(ascending=False).head(15))
print(C.groupby('activity')[['std','rms','dom_freq','tot_std','raw_diff_rms','hf_frac','pca1','acf_peak']].median().round(3))
print(S.groupby('activity')[['std','rms','dom_freq','tot_std','raw_diff_rms','hf_frac','pca1','acf_peak']].median().round(3))
