import sys; sys.path.insert(0,'research')
from evalkit import *
L=5
C=pd.read_pickle(f'research/cache2_Class_Data_{L}.pkl').fillna(0)
S=pd.read_pickle(f'research/cache2_Sample_Data_{L}.pkl').fillna(0)
allf=[c for c in C.columns if c not in('activity','recording')]
for nm in ['rf','et']:
    print(nm,'split',round(split(C,allf,nm),3),'C->S',[round(float(v),3) for v in xdom(C,S,allf,nm)],'S->C',[round(float(v),3) for v in xdom(S,C,allf,nm)])
D=C[C.activity.isin(['lying','sitting','standing'])]
m=mk('et').fit(D[allf],D.activity)
print(pd.Series(m.feature_importances_,allf).sort_values(ascending=False).head(20).round(3))
print(D.groupby('activity')[['mean_x','mean_y','mean_z','vf_x','vf_y','vf_z','lfr_x','lfr_y','lfr_z','hfr_x','hfr_y','hfr_z']].median().round(3))
