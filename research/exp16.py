import sys; sys.path.insert(0,'research')
from evalkit import *
from sklearn.ensemble import HistGradientBoostingClassifier
import evalkit
_mk=evalkit.mk
def mk2(n):
    if n=='hgb': return HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,random_state=0)
    return _mk(n)
evalkit.mk=mk2
C=pd.read_pickle('research/cache3_Class_Data_5.pkl').replace([np.inf,-np.inf],0).fillna(0)
S=pd.read_pickle('research/cache3_Sample_Data_5.pkl').replace([np.inf,-np.inf],0).fillna(0)
sel=['min', 'v_lagmax', 'v_std', 'lf_hf', 'v_d2', 'lf_z', 'vf_y', 'kurt', 'mean_z', 'acf_peak', 'mean_y', 'vf_z', 'v_kurt', 'logvar', 'pca2', 'v_skew', 'lfr_y', 'lfr_z', 'med', 'c_yz']
sets={'top8':sel[:8],'top10':sel[:10],'top12':sel[:12],'top16':sel[:16],'top20':sel,
 'old8':['min','tot_std','raw_diff_rms','hf_frac','peaks_ps','acf_lag','spec_ent','bp_0.3_1'],
 'all':[c for c in C.columns if c not in('activity','recording')]}
for k,fs in sets.items():
    for nm in ['dt5','rf','hgb']:
        print(k,nm,'split %.3f'%split(C,fs,nm),'gcv',[round(float(v),3) for v in gcv(C,fs,nm)],'C->S',[round(float(v),3) for v in xdom(C,S,fs,nm)],'S->C',[round(float(v),3) for v in xdom(S,C,fs,nm)],flush=True)
