import os, glob, numpy as np, pandas as pd, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis
from featlib import cand_features, est_fs, load
from util import calc_magnitude, remove_noise

def bp(a, fs, lo=None, hi=None):
    if lo and hi: b, c = butter(2, [lo/(fs/2), hi/(fs/2)], 'band')
    elif hi: b, c = butter(2, hi/(fs/2), 'low')
    else: b, c = butter(2, lo/(fs/2), 'high')
    return filtfilt(b, c, a, axis=0)

def extra(w, fs, lf, hf):
    F = {}
    xyz = w[['x','y','z']].values
    tot = xyz.var(0).sum() + 1e-12
    for i, n in enumerate('xyz'):
        F[f'vf_{n}'] = xyz[:, i].var() / tot
        F[f'lf_{n}'] = lf[:, i].std(); F[f'hf_{n}'] = hf[:, i].std()
        F[f'mean_{n}'] = xyz[:, i].mean()
    lft = np.sqrt((lf.var(0)).sum()) + 1e-12; hft = np.sqrt(hf.var(0).sum()) + 1e-12
    for i, n in enumerate('xyz'):
        F[f'lfr_{n}'] = lf[:, i].std() / lft; F[f'hfr_{n}'] = hf[:, i].std() / hft
    F['lf_tot'] = lft; F['hf_tot'] = hft; F['lf_hf'] = lft / hft
    # vertical proxy: first principal axis of the low-passed motion
    c = xyz - xyz.mean(0)
    sm = bp(c, fs, hi=5) if len(c) > 15 else c
    ev, evec = np.linalg.eigh(np.cov(sm.T))
    v = sm @ evec[:, -1]
    if abs(v.min()) > abs(v.max()): v = -v
    h = sm - np.outer(v, evec[:, -1])
    hm = np.linalg.norm(h, axis=1)
    F['v_skew'] = skew(v); F['v_kurt'] = kurtosis(v); F['v_std'] = v.std(); F['h_std'] = hm.std() + np.sqrt((h**2).sum(1).mean())
    F['vh_ratio'] = v.std() / (np.sqrt((h**2).sum(1).mean()) + 1e-9)
    F['v_absskew'] = abs(skew(v))
    F['v_p95_p5'] = (np.percentile(v, 95)) / (abs(np.percentile(v, 5)) + 1e-9)
    F['vh_corr'] = abs(np.corrcoef(v, hm)[0, 1]) if hm.std() > 0 else 0
    # jerk of vertical
    dv = np.diff(v) * fs
    F['v_jerk_skew'] = skew(dv); F['v_jerk_kurt'] = kurtosis(dv)
    return F

def windows_for2(root, L):
    rows = []
    for f in sorted(glob.glob(f'{root}/*/*.csv')):
        d = load(f); act = os.path.basename(os.path.dirname(f)); fs = est_fs(d['time'])
        d = calc_magnitude(d); d = remove_noise(d, fs)
        xyz = d[['x','y','z']].values
        lf = bp(xyz, fs, hi=1.0); hf = bp(xyz, fs, lo=min(5, fs/2*0.8)) if fs > 12 else xyz
        d[['lfx','lfy','lfz']] = lf; d[['hfx','hfy','hfz']] = hf
        for ts, w in d.set_index('time').resample(f'{L}s'):
            if len(w) < 0.8 * L * fs: continue
            F = cand_features(w, fs); F.update(extra(w, fs, w[['lfx','lfy','lfz']].values, w[['hfx','hfy','hfz']].values))
            F['activity'] = act; F['recording'] = act + '/' + os.path.basename(f); rows.append(F)
    return pd.DataFrame(rows)
if __name__ == '__main__':
    from joblib import Parallel, delayed
    def job(r, L): windows_for2(f'data/{r}', L).to_pickle(f'research/cache2_{r}_{L}.pkl'); return r, L
    print(Parallel(n_jobs=6)(delayed(job)(r, L) for r in ['Sample_Data', 'Class_Data'] for L in [5, 10]))
