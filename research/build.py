import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featlib import *
from joblib import Parallel, delayed
import time
def job(root, L):
    t = time.time(); df = windows_for(f'data/{root}', L)
    df.to_pickle(f'research/cache_{root}_{L}.pkl'); return root, L, len(df), time.time() - t
res = Parallel(n_jobs=8)(delayed(job)(r, L) for r in ['Sample_Data', 'Class_Data'] for L in [3, 5, 10])
for r in res: print(r)
