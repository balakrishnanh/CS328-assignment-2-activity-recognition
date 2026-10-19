import sys; sys.path.insert(0,'research')
from evalkit import *
from sklearn.ensemble import HistGradientBoostingClassifier
import evalkit
cfg={'h1':dict(max_iter=300,learning_rate=0.05),'h2':dict(max_iter=400,learning_rate=0.03,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=1.0),
'h3':dict(max_iter=200,learning_rate=0.05,max_depth=4,min_samples_leaf=50),'h4':dict(max_iter=600,learning_rate=0.02,max_leaf_nodes=31,min_samples_leaf=30,l2_regularization=1.0,max_features=0.5)}
_mk=evalkit.mk
evalkit.mk=lambda n: HistGradientBoostingClassifier(random_state=0,**cfg[n]) if n in cfg else (ExtraTreesClassifier(500,min_samples_leaf=2,max_features=0.5,random_state=0,n_jobs=-1) if n=='et5' else _mk(n))
C=pd.read_csv('research/v_class_5s.csv');S=pd.read_csv('research/v_sample_5s.csv')
for d in (C,S): d['recording']=d.activity+'/'+d.recording
top20=['min','vert_period','vert_std','slow_fast_ratio','vert_stride_reg','slow_z','var_frac_y','kurt','mean_z','acf_peak','mean_y','var_frac_z','vert_kurt','logvar','pca2','vert_skew','slow_frac_y','slow_frac_z','med','corr_yz']
for nm in ['h2','h3','h4','et','et5']:
    print(nm,'split %.3f'%split(C,top20,nm),'gcv',[round(float(v),3) for v in gcv(C,top20,nm)],'C->S',[round(float(v),3) for v in xdom(C,S,top20,nm)],flush=True)
