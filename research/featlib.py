import os, glob, numpy as np, pandas as pd, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from util import calc_magnitude, remove_noise, add_features

def est_fs(t):
    dt = np.diff(t.values.astype('datetime64[ns]').astype('int64')) / 1e9
    dt = dt[dt > 0]
    return float(1.0 / np.median(dt))

def spec(sig, fs, nfft=1024):
    s = sig - sig.mean()
    s = s * np.hanning(len(s))
    P = np.abs(np.fft.rfft(s, nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / fs)
    return f, P

def cand_features(w, fs):
    m = w['filtered_accel_mag'].values
    x, y, z = w['x'].values, w['y'].values, w['z'].values
    F = add_features(w).iloc[0].to_dict()
    mc = m - m.mean()
    F['rms'] = float(np.sqrt(np.mean(m ** 2)))
    F['iqr'] = F['q75'] - F['q25']
    F['range'] = F['max'] - F['min']
    F['zcr'] = float(np.sum(np.diff(np.sign(mc)) != 0) / (len(m) / fs))
    F['jerk'] = float(np.mean(np.abs(np.diff(m))) * fs)
    pk, _ = find_peaks(mc, distance=max(1, int(0.25 * fs)), prominence=0.5 * (mc.std() + 1e-9))
    F['peaks_ps'] = len(pk) / (len(m) / fs)
    f, P = spec(m, fs)
    band = (f >= 0.5) & (f <= 5)
    tot = P[band].sum() + 1e-12
    i = np.argmax(np.where(band, P, -1))
    F['dom_freq'] = float(f[i])
    F['dom_ratio'] = float(P[i] / tot)
    for a, b in [(0.3, 1), (1, 2), (2, 3), (3, 5)]:
        bb = (f >= a) & (f < b)
        F[f'bp_{a}_{b}'] = float(P[bb].sum() / (P[(f >= 0.3) & (f <= 5)].sum() + 1e-12))
    pn = P[(f >= 0.3) & (f <= 5)]; pn = pn / (pn.sum() + 1e-12)
    F['spec_ent'] = float(-(pn * np.log(pn + 1e-12)).sum() / np.log(len(pn)))
    # autocorrelation periodicity
    ac = np.correlate(mc, mc, 'full')[len(mc) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(0.25 * fs), min(len(ac) - 1, int(1.5 * fs))
    if hi > lo:
        j = lo + np.argmax(ac[lo:hi]); F['acf_peak'] = float(ac[j]); F['acf_lag'] = j / fs
    else:
        F['acf_peak'] = 0.0; F['acf_lag'] = 0.0
    F['skew'] = float(skew(m)); F['kurt'] = float(kurtosis(m))
    for n, a in (('x', x), ('y', y), ('z', z)):
        F[f'std_{n}'] = float(a.std())
    C = np.cov(np.vstack([x, y, z]))
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]; s = ev.sum() + 1e-12
    F['pca1'] = float(ev[0] / s); F['pca2'] = float(ev[1] / s); F['pca3'] = float(ev[2] / s)
    F['tot_std'] = float(np.sqrt(max(s, 0)))
    F['raw_diff_rms'] = float(np.sqrt(np.mean(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)))
    F['mean_norm'] = float(np.sqrt(x.mean() ** 2 + y.mean() ** 2 + z.mean() ** 2))
    rm = w['accel_mag'].values
    fr, Pr = spec(rm, fs)
    F['hf_frac'] = float(Pr[fr > 5].sum() / (Pr.sum() + 1e-12))
    F['logvar'] = float(np.log(np.var(m) + 1e-8))
    return F

def load(f):
    d = pd.read_csv(f)
    t = d['time']
    d['time'] = pd.to_datetime(t, unit='ns') if np.issubdtype(t.dtype, np.number) else pd.to_datetime(t)
    return d

_cache = {}
def windows_for(root, L, trim=0.0):
    key = (root, L, trim)
    if key in _cache: return _cache[key]
    rows = []
    for f in sorted(glob.glob(f'{root}/*/*.csv')):
        d = load(f); act = os.path.basename(os.path.dirname(f))
        fs = est_fs(d['time'])
        d = calc_magnitude(d); d = remove_noise(d, fs)
        if trim:
            t0 = d['time'].iloc[0] + pd.Timedelta(seconds=trim); t1 = d['time'].iloc[-1] - pd.Timedelta(seconds=trim)
            d = d[(d['time'] >= t0) & (d['time'] <= t1)]
        t00 = d['time'].iloc[0]; dur = (d['time'].iloc[-1] - t00).total_seconds()
        for ts, w in d.set_index('time').resample(f'{L}s'):
            if len(w) < 0.8 * L * fs: continue
            F = cand_features(w, fs); F['activity'] = act; F['recording'] = os.path.basename(f)
            F['t_start'] = (w.index[0] - t00).total_seconds(); F['rec_dur'] = dur; F['fs'] = fs; rows.append(F)
    df = pd.DataFrame(rows); _cache[key] = df; return df
