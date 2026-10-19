import sys; sys.path.insert(0,'research')
from evalkit import *
from sklearn.ensemble import HistGradientBoostingClassifier
import evalkit
_mk=evalkit.mk
def mk2(n):
    if n=='hgb': return HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,random_state=0)
    return _mk(n)
evalkit.mk=mk2
C=pd.read_csv('research/v_class_5s.csv');S=pd.read_csv('research/v_sample_5s.csv')
for d in (C,S): d['recording']=d.activity+'/'+d.recording
top20=['min','vert_period','vert_std','slow_fast_ratio','vert_stride_reg','slow_z','var_frac_y','kurt','mean_z','acf_peak','mean_y','var_frac_z','vert_kurt','logvar','pca2','vert_skew','slow_frac_y','slow_frac_z','med','corr_yz']
allf=[c for c in C.columns if c not in('activity','recording')]
for k,fs in {'top8':top20[:8],'top12':top20[:12],'top20':top20,'all46':allf}.items():
    for nm in ['rf','hgb']:
        print(k,nm,'split %.3f'%split(C,fs,nm),'gcv',[round(float(v),3) for v in gcv(C,fs,nm)],'C->S',[round(float(v),3) for v in xdom(C,S,fs,nm)],'S->C',[round(float(v),3) for v in xdom(S,C,fs,nm)],flush=True)
