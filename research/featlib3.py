import os, glob, numpy as np, pandas as pd, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.signal import find_peaks
from scipy.stats import skew
from featlib2 import bp, extra
from featlib import cand_features, est_fs, load
from util import calc_magnitude, remove_noise

def acf(s):
    s = s - s.mean(); a = np.correlate(s, s, 'full')[len(s)-1:]; return a / (a[0] + 1e-12)

def gait(w, fs):
    F = {}
    xyz = w[['x','y','z']].values; c = xyz - xyz.mean(0)
    sm = bp(c, fs, hi=5)
    ev, evec = np.linalg.eigh(np.cov(sm.T)); v = sm @ evec[:, -1]
    h = np.linalg.norm(sm - np.outer(v, evec[:, -1]), axis=1)
    for nm, s in (('v', v), ('h', h)):
        a = acf(s); lo = int(0.3 * fs); hi = min(len(a) - 1, int(2.4 * fs))
        pk, pr = find_peaks(a[lo:hi], height=-1)
        if len(pk):
            order = pk[np.argsort(-a[lo:hi][pk])]
            p1 = pk[0]; F[f'{nm}_d1'] = a[lo + p1]; F[f'{nm}_lag1'] = (lo + p1) / fs
            F[f'{nm}_dmax'] = a[lo + order[0]]; F[f'{nm}_lagmax'] = (lo + order[0]) / fs
            F[f'{nm}_d2'] = a[lo + pk[1]] if len(pk) > 1 else 0
            F[f'{nm}_sym'] = F[f'{nm}_d1'] / (abs(F[f'{nm}_d2']) + 1e-3) if len(pk) > 1 else 0
        else:
            for k in ('d1','lag1','dmax','lagmax','d2','sym'): F[f'{nm}_{k}'] = 0
        # harmonic ratio over spectrum
        n = max(1024, 2 ** int(np.ceil(np.log2(len(s)))))
        P = np.abs(np.fft.rfft((s - s.mean()) * np.hanning(len(s)), n)) ** 2; f = np.fft.rfftfreq(n, 1/fs)
        g = (f >= 0.6) & (f <= 3.5); f0 = f[g][np.argmax(P[g])] if g.any() else 1
        F[f'{nm}_f0'] = f0
        e1 = P[(f > f0*0.9) & (f < f0*1.1)].sum(); e2 = P[(f > 2*f0*0.9) & (f < 2*f0*1.1)].sum(); eh = P[(f > f0*0.5*0.9) & (f < f0*0.5*1.1)].sum()
        F[f'{nm}_h2'] = e2 / (e1 + 1e-12); F[f'{nm}_hhalf'] = eh / (e1 + 1e-12)
    for i, n in enumerate('xyz'):
        a = acf(c[:, i]); lo = int(0.3 * fs); hi = min(len(a) - 1, int(2.4 * fs)); F[f'acfmax_{n}'] = a[lo:hi].max() if hi > lo else 0
    cc = np.corrcoef(c.T); F['c_xy'], F['c_xz'], F['c_yz'] = cc[0, 1], cc[0, 2], cc[1, 2]
    return F

def windows_for3(root, L, step=None):
    rows = []
    for f in sorted(glob.glob(f'{root}/*/*.csv')):
        d = load(f); act = os.path.basename(os.path.dirname(f)); fs = est_fs(d['time'])
        d = calc_magnitude(d); d = remove_noise(d, fs)
        xyz = d[['x','y','z']].values
        lf = bp(xyz, fs, hi=1.0); hf = bp(xyz, fs, lo=min(5, fs/2*0.8))
        d[['lfx','lfy','lfz']] = lf; d[['hfx','hfy','hfz']] = hf
        for ts, w in d.set_index('time').resample(f'{L}s'):
            if len(w) < 0.8 * L * fs: continue
            F = cand_features(w, fs); F.update(extra(w, fs, w[['lfx','lfy','lfz']].values, w[['hfx','hfy','hfz']].values)); F.update(gait(w, fs))
            F['activity'] = act; F['recording'] = act + '/' + os.path.basename(f); rows.append(F)
    return pd.DataFrame(rows)
if __name__ == '__main__':
    from joblib import Parallel, delayed
    def job(r, L): windows_for3(f'data/{r}', L).to_pickle(f'research/cache3_{r}_{L}.pkl'); return r, L
    print(Parallel(n_jobs=6)(delayed(job)(r, L) for r in ['Sample_Data', 'Class_Data'] for L in [5, 10]))
