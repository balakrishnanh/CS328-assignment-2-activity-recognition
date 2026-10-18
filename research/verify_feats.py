import sys, time; sys.path.insert(0,'research'); sys.path.insert(0,'.')
import cells
src = cells.IMPORTS.replace('%matplotlib inline','') + cells.EXTRACT + cells.COMBINE + cells.MYFEATS
g = {}; exec(src, g)
from joblib import Parallel, delayed
import pandas as pd
t=time.time()
C = g['all_data_to_combined_csv']('./data/Class_Data', 'research/v_class_5s.csv', 5); S = g['all_data_to_combined_csv']('./data/Sample_Data', 'research/v_sample_5s.csv', 5)
print('built', len(C), len(S), C.shape[1], round(time.time()-t))
