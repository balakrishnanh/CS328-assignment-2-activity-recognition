import sys; sys.path.insert(0,'research')
from evalkit import *
from featlib import load
import glob,os
pd.set_option('display.width',250)
# per-recording stats for static classes in Class_Data
rows=[]
for f in glob.glob('data/Class_Data/*/*.csv'):
    a=os.path.basename(os.path.dirname(f))
    if a not in('lying','sitting','standing'):continue
    d=load(f);x,y,z=d.x.values,d.y.values,d.z.values
    n=len(d)//2
    rows.append(dict(a=a,mx=x.mean(),my=y.mean(),mz=z.mean(),sx=x.std(),sy=y.std(),sz=z.std(),mn=np.sqrt(x.mean()**2+y.mean()**2+z.mean()**2),tot=np.sqrt(x.var()+y.var()+z.var()),dur=len(d)/100))
r=pd.DataFrame(rows)
print(r.groupby('a').describe().T.loc[(slice(None),['25%','50%','75%']),:].round(3).to_string())
