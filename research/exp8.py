import sys; sys.path.insert(0,'research')
from evalkit import *
sets={'A':['min','acf_lag','std_z','peaks_ps','spec_ent','hf_frac','bp_0.3_1','pca2'],
'B':['min','acf_lag','tot_std','peaks_ps','spec_ent','hf_frac','bp_0.3_1','raw_diff_rms'],
'B2':['min','std','acf_lag','tot_std','peaks_ps','spec_ent','hf_frac','dom_freq'],
'C':BASE+['rms','dom_freq','acf_peak','tot_std','hf_frac'],
'base':BASE}
for L in [5,10]:
    S,C=ld('Sample_Data',L),ld('Class_Data',L)
    sets['all']=[c for c in C.columns if c not in('activity','recording','t_start','rec_dur','fs')]
    for k,fs in sets.items():
        for name in ['dt5','et','svm']:
            g=gcv(C,fs,name);cs=xdom(C,S,fs,name);sc=xdom(S,C,fs,name)
            print(L,k,name,'split %.3f gcv %.3f/%.3f C->S %.3f/%.3f S->C %.3f/%.3f'%(split(C,fs,name),*g,*cs,*sc),flush=True)
