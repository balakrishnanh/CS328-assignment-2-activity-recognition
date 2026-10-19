import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupKFold
BASE = ['avg','max','med','min','q25','q75','std']
def ld(root, L):
    d = pd.read_pickle(f'research/cache_{root}_{L}.pkl'); d['recording'] = d.activity + '/' + d.recording; return d
def mk(name):
    return {'dt5': lambda: DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42),
     'rf': lambda: RandomForestClassifier(300, random_state=0, n_jobs=-1),
     'et': lambda: ExtraTreesClassifier(300, random_state=0, n_jobs=-1),
     'knn': lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
     'lr': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000)),
     'svm': lambda: make_pipeline(StandardScaler(), SVC(C=3)),
     'gb': lambda: GradientBoostingClassifier(random_state=0)}[name]()
def recacc(df, pred):
    t = df.assign(p=pred).groupby('recording').agg(true=('activity','first'), p=('p', lambda s: s.value_counts().idxmax()))
    return (t.true == t.p).mean()
def split(df, feats, name):
    tr, te = train_test_split(df, test_size=0.3, random_state=42)
    m = mk(name).fit(tr[feats], tr.activity); return m.score(te[feats], te.activity)
def gcv(df, feats, name, k=5):
    accs=[]; P=np.empty(len(df), dtype=object)
    for a,b in GroupKFold(k).split(df, groups=df.recording):
        m = mk(name).fit(df.iloc[a][feats], df.iloc[a].activity); P[b] = m.predict(df.iloc[b][feats])
    return (P==df.activity.values).mean(), recacc(df,P)
def xdom(tr, te, feats, name):
    m = mk(name).fit(tr[feats], tr.activity); p = m.predict(te[feats])
    return (p==te.activity.values).mean(), recacc(te,p)
