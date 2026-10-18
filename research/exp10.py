import sys; sys.path.insert(0,'research')
from evalkit import *
for L in [5,10]:
    C=pd.read_pickle(f'research/cache2_Class_Data_{L}.pkl').fillna(0)
    meta=('activity','recording')
    allf=[c for c in C.columns if c not in meta]
    old=[c for c in allf if not any(c.startswith(p) for p in ('vf_','lf','hf_','hfr_','mean_','v_','h_','vh_'))] 
    old=[c for c in old if c!='hf_tot' and c!='lf_tot' and c!='lf_hf']+['hf_frac']
    old=list(dict.fromkeys(old))
    axis=[c for c in allf if c[:3] in('vf_','lf_','hf_','lfr','hfr','mea')]+['lf_tot','hf_tot','lf_hf']
    vert=[c for c in allf if c.startswith(('v_','vh_','h_'))]
    groups={'static':['lying','sitting','standing'],'gait':['walk','upstairs','downstairs']}
    for g,cl in groups.items():
        D=C[C.activity.isin(cl)].reset_index(drop=True)
        for nm,fs in [('old',old),('old+axis',old+axis),('old+vert',old+vert),('all',allf)]:
            print(L,g,nm,len(fs),[round(float(v),3) for v in gcv(D,fs,'et')],flush=True)
    print(L,'7-class all',[round(float(v),3) for v in gcv(C,allf,'et')])
