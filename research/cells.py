"""Source text for the code cells of A2_activity_recognition.ipynb, keyed by cell index."""

IMPORTS = r'''# -- IMPORTS START --
import pandas as pd
import glob
import re
import os
import sys
import json
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import skew, kurtosis
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from util import viz_tree, calc_magnitude, remove_noise, add_features
# -- IMPORTS END --

# enable zooming into graphs
%matplotlib inline
plt.rcParams['figure.figsize'] = [9, 6] # width, height in inches
os.makedirs('figures', exist_ok=True)
RESULTS = {}   # every number quoted in the exploration report is stored here and written to results.json
'''

EXTRACT = r'''# Function to extract windows and features
def extract_features(data, window_sec, sample_rate, activity):

    ######## TODO ###############
    # Non-overlapping windows of window_sec seconds (resample needs the time as index)
    min_samples = 0.8 * window_sec * sample_rate
    rows = []
    for _, window in data.set_index('time').resample(f'{window_sec}s'):
        # partial windows at the start/end of a recording, or gaps in the logging
        if len(window) < min_samples:
            continue
        features = add_my_features(window, sample_rate)
        features['activity'] = activity
        rows.append(features)
    frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    #############################

    return frame
'''

COMBINE = r'''def all_data_to_combined_csv(root, output_filename = 'all_data.csv', window_sec = 5):
    import os
    import sys

    ######## TODO ###############
    files = sorted(glob.glob(os.path.join(root, '*', '*.csv')))
    frames = []
    for filename in files:
        data = pd.read_csv(filename, parse_dates=['time'])
        if pd.api.types.is_numeric_dtype(data['time']):      # file not converted yet (ns since 1970)
            data['time'] = pd.to_datetime(data['time'], unit='ns')
        # column order differs between the two folders, so select by name
        data = data[['time', 'x', 'y', 'z']].dropna().sort_values('time').reset_index(drop=True)

        # sampling rate from the median timestamp step (most phones ~100 Hz, a few 50-60 Hz)
        dt = data['time'].diff().dt.total_seconds()
        sample_rate = 1.0 / dt[dt > 0].median()

        data = calc_magnitude(data)
        data = remove_noise(data, sample_rate)

        activity = os.path.basename(os.path.dirname(filename))
        frame = extract_features(data, window_sec, sample_rate, activity)
        if frame.empty:          # recording shorter than one window
            continue
        frame['recording'] = os.path.basename(filename)
        frames.append(frame)

    all_data = pd.concat(frames, ignore_index=True)
    all_data.to_csv(output_filename, index=False)
    #############################

    return all_data
'''

MYFEATS = r'''def add_my_features(window, sample_rate):
    features = add_features(window)   # baseline: avg, max, med, min, q25, q75, std of filtered_accel_mag

    ######## TODO ###############
    mag = window['filtered_accel_mag'].to_numpy()        # low-passed (5 Hz) magnitude
    raw_mag = window['accel_mag'].to_numpy()             # unfiltered magnitude
    xyz = window[['x', 'y', 'z']].to_numpy()
    centered = mag - mag.mean()
    seconds = len(mag) / sample_rate

    # --- time domain ---
    features['rms'] = np.sqrt(np.mean(mag ** 2))
    features['zcr'] = np.sum(np.diff(np.sign(centered)) != 0) / seconds
    # steps: peaks at least 0.25 s apart that stand out by half a standard deviation
    peaks, _ = find_peaks(centered, distance=max(1, int(0.25 * sample_rate)),
                          prominence=0.5 * centered.std() + 1e-9)
    features['peaks_per_sec'] = len(peaks) / seconds

    # --- step periodicity: lag of the strongest autocorrelation peak between 0.25 s and 1.5 s ---
    ac = np.correlate(centered, centered, mode='full')[len(centered) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(0.25 * sample_rate), min(len(ac) - 1, int(1.5 * sample_rate))
    lag = lo + int(np.argmax(ac[lo:hi])) if hi > lo else 0
    features['step_period'] = lag / sample_rate
    features['acf_peak'] = ac[lag] if hi > lo else 0.0

    # --- frequency domain (Hann window, zero-padded FFT) ---
    nfft = max(1024, 2 ** int(np.ceil(np.log2(len(mag)))))
    freqs = np.fft.rfftfreq(nfft, 1 / sample_rate)
    power = np.abs(np.fft.rfft(centered * np.hanning(len(centered)), nfft)) ** 2
    gait = (freqs >= 0.5) & (freqs <= 5)
    features['dom_freq'] = freqs[gait][np.argmax(power[gait])]
    band = (freqs >= 0.3) & (freqs <= 5)
    p = power[band] / (power[band].sum() + 1e-12)
    features['spec_entropy'] = -(p * np.log(p + 1e-12)).sum() / np.log(len(p))
    features['low_band_frac'] = power[(freqs >= 0.3) & (freqs < 1)].sum() / (power[band].sum() + 1e-12)
    # share of the unfiltered magnitude's power above the 5 Hz cutoff (tremor, sensor noise, impacts)
    raw_c = raw_mag - raw_mag.mean()
    raw_power = np.abs(np.fft.rfft(raw_c * np.hanning(len(raw_c)), nfft)) ** 2
    features['hf_frac'] = raw_power[freqs > 5].sum() / (raw_power.sum() + 1e-12)

    # --- per axis and orientation-free 3-D motion ---
    for i, axis in enumerate('xyz'):
        features[f'std_{axis}'] = xyz[:, i].std()
    eig = np.clip(np.linalg.eigvalsh(np.cov(xyz.T)), 0, None)
    features['axis_std'] = np.sqrt(eig.sum())                      # total 3-D motion, rotation invariant
    features['pca1'] = eig.max() / (eig.sum() + 1e-12)             # how one-directional the motion is
    features['jerk_raw'] = np.sqrt(np.mean(np.sum(np.diff(xyz, axis=0) ** 2, axis=1))) * sample_rate
    features['pca2'] = np.sort(eig)[1] / (eig.sum() + 1e-12)       # flat (2-D) vs line-like vs isotropic motion

    # --- shape of the magnitude distribution ---
    features['kurt'] = kurtosis(mag)                               # spiky impacts vs smooth motion
    features['logvar'] = np.log(np.var(mag) + 1e-8)                # intensity on a log scale (rest spans decades)

    # --- what is left of orientation. Sensor Logger's gravity removal is not perfect: the residual
    #     mean and the slow (<1 Hz) drift land on the axes that are not vertical, so the per-axis
    #     split of slow motion still carries some information about how the phone is held. ---
    def butter_filter(sig, cutoff, btype):
        b, a = butter(2, cutoff / (sample_rate / 2), btype=btype)
        return filtfilt(b, a, sig, axis=0)
    slow = butter_filter(xyz, 1.0, 'lowpass')
    fast = butter_filter(xyz, min(5.0, 0.4 * sample_rate), 'highpass')
    slow_tot = np.sqrt(slow.var(axis=0).sum()) + 1e-12
    fast_tot = np.sqrt(fast.var(axis=0).sum()) + 1e-12
    var_tot = xyz.var(axis=0).sum() + 1e-12
    for i, axis in enumerate('xyz'):
        features[f'mean_{axis}'] = xyz[:, i].mean()                # residual gravity along the axis
        features[f'var_frac_{axis}'] = xyz[:, i].var() / var_tot   # share of motion on the axis
        features[f'slow_{axis}'] = slow[:, i].std()
        features[f'slow_frac_{axis}'] = slow[:, i].std() / slow_tot
    features['slow_fast_ratio'] = slow_tot / fast_tot              # sway/drift vs tremor and sensor noise
    with np.errstate(invalid='ignore', divide='ignore'):
        features['corr_yz'] = np.corrcoef(xyz[:, 1], xyz[:, 2])[0, 1]

    # --- vertical proxy: project the (5 Hz low-passed) motion on its main axis. While walking the
    #     main axis is close to vertical, whatever way the phone is rotated. Sign: largest excursion up. ---
    smooth = butter_filter(xyz - xyz.mean(axis=0), min(5.0, 0.4 * sample_rate), 'lowpass')
    w, v = np.linalg.eigh(np.cov(smooth.T))
    vert = smooth @ v[:, -1]
    if abs(vert.min()) > abs(vert.max()):
        vert = -vert
    features['vert_std'] = vert.std()
    features['vert_skew'] = skew(vert)                             # push-off (up) vs landing (down) asymmetry
    features['vert_kurt'] = kurtosis(vert)
    # gait regularity (Moe-Nilssen): autocorrelation peaks of the vertical signal between 0.3 s and 2.4 s
    vc = vert - vert.mean()
    vac = np.correlate(vc, vc, mode='full')[len(vc) - 1:]
    vac = vac / (vac[0] + 1e-12)
    lo, hi = int(0.3 * sample_rate), min(len(vac) - 1, int(2.4 * sample_rate))
    pk, _ = find_peaks(vac[lo:hi], height=-1) if hi > lo else (np.array([], dtype=int), None)
    features['vert_period'] = (lo + pk[np.argmax(vac[lo:hi][pk])]) / sample_rate if len(pk) else 0.0
    features['vert_stride_reg'] = vac[lo + pk[1]] if len(pk) > 1 else 0.0   # 2nd peak = one full stride

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    #############################

    return features
'''

TIME = None  # cell 11 left as provided

HELPERS_EXTRA = None

FEATURE_EXPLORATION = r'''# Hints, step by step:
# Build the class corpus with different window lengths, e.g. all_data_to_combined_csv('./data/Class_Data', 'class_2s.csv', window_sec=2)
# Train train_decision_tree on each with features=BASELINE_FEATURES and compare the accuracy
# Write add_my_features, rebuild, and compare baseline vs your features at the same window length
# Lock in WINDOW_SEC and FEATURES for the rest of the notebook

########################### TODO ##########################
CLASSES = ['walk', 'upstairs', 'downstairs', 'jogging', 'standing', 'sitting', 'lying']

def split_scores(frames, features, max_depth=5, criterion='entropy'):
    """Same 70/30 split and tree as train_decision_tree, without the printout."""
    X_tr, X_te, y_tr, y_te = train_test_split(frames[features], frames['activity'], test_size=0.3, random_state=42)
    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    f1 = f1_score(y_te, pred, labels=CLASSES, average=None, zero_division=0)
    return accuracy_score(y_te, pred), dict(zip(CLASSES, f1))

def grouped_cv(frames, features, make_model, k=5):
    """Honest check: whole recordings are held out, so no recording is in both train and test."""
    pred = np.empty(len(frames), dtype=object)
    groups = frames['activity'] + '/' + frames['recording']
    for tr, te in GroupKFold(k).split(frames, groups=groups):
        model = make_model().fit(frames.iloc[tr][features], frames.iloc[tr]['activity'])
        pred[te] = model.predict(frames.iloc[te][features])
    return accuracy_score(frames['activity'], pred), pred

# 1. Window length, baseline features
class_frames = {}
window_rows = []
for w in [2, 3, 5, 10]:
    class_frames[w] = all_data_to_combined_csv('./data/Class_Data', f'class_{w}s.csv', window_sec=w)
    print(f'===== {w} s windows, baseline features ({len(class_frames[w])} windows) =====')
    _, _, acc = train_decision_tree(class_frames[w], features=BASELINE_FEATURES)
    cv_acc, _ = grouped_cv(class_frames[w], BASELINE_FEATURES,
                           lambda: DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42))
    window_rows.append({'window_sec': w, 'windows': len(class_frames[w]),
                        'recordings': class_frames[w].groupby(['activity', 'recording']).ngroups,
                        'split_acc': acc, 'grouped_cv_acc': cv_acc})
window_table = pd.DataFrame(window_rows)
print(window_table.round(3))
RESULTS['window_baseline'] = window_rows
'''

FEATURE_EXPLORATION2 = r'''# 2. Feature engineering at 5 s. All candidate columns come out of add_my_features.
WINDOW_SEC = 5
frames5 = class_frames[WINDOW_SEC]
CANDIDATES = [c for c in feature_columns(frames5) if c not in BASELINE_FEATURES]
print('candidate features:', CANDIDATES)

# Final set (locked in below): 20 features, chosen by greedy forward selection with whole recordings held out
# (research/exp15.py, extra trees). Order = order of selection. The first 8 alone are the compact variant
# (FEATURES[:8], see the table): it gives up a few points of grouped-CV accuracy for a much smaller model.
FEATURES = ['min', 'vert_period', 'vert_std', 'slow_fast_ratio', 'vert_stride_reg', 'slow_z', 'var_frac_y', 'kurt',
            'mean_z', 'acf_peak', 'mean_y', 'var_frac_z', 'vert_kurt', 'logvar', 'pca2', 'vert_skew',
            'slow_frac_y', 'slow_frac_z', 'med', 'corr_yz']
OLD8 = ['min', 'axis_std', 'jerk_raw', 'hf_frac', 'peaks_per_sec', 'step_period', 'spec_entropy', 'low_band_frac']

rf = lambda: RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1)
dt = lambda: DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)

def feature_row(name, feats):
    acc, f1 = split_scores(frames5, feats)
    cv_dt, _ = grouped_cv(frames5, feats, dt)
    cv_rf, _ = grouped_cv(frames5, feats, rf)
    return {'set': name, 'n': len(feats), 'dt_split': acc, 'dt_grouped_cv': cv_dt, 'rf_grouped_cv': cv_rf, **{f'f1_{c}': f1[c] for c in CLASSES}}

rows = [feature_row('baseline (7)', BASELINE_FEATURES)]
for f in CANDIDATES:                                  # baseline + one candidate
    rows.append(feature_row(f'baseline + {f}', BASELINE_FEATURES + [f]))
rows.append(feature_row('first-round 8 (magnitude only)', OLD8))
rows.append(feature_row('compact 8 = FEATURES[:8]', FEATURES[:8]))
rows.append(feature_row('compact 12 = FEATURES[:12]', FEATURES[:12]))
rows.append(feature_row('FINAL (20)', FEATURES))
for f in FEATURES:                                    # final minus one feature (ablation)
    rows.append(feature_row(f'final - {f}', [g for g in FEATURES if g != f]))
rows.append(feature_row('baseline + all candidates', BASELINE_FEATURES + CANDIDATES))
feature_table = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(feature_table.round(3).to_string())
RESULTS['feature_table'] = rows

print('\n===== Locked in: 5 s windows, FEATURES =====')
dt_final, cm_final, acc_final = train_decision_tree(frames5, features=FEATURES)
RESULTS['window_final'] = {'window_sec': WINDOW_SEC, 'features': FEATURES, 'split_acc': acc_final}
'''

CLASS1 = r'''# Hints, step by step:
# Combine all the data under data/Sample_Data
# Combine all the data under data/Class_Data
# Train the decision tree
# Grid search, choosing by the Sample_Data test split accuracy
# Display the confusion matrix
# Visualize the tree
# Evaluate the model on class dataset

########################### TODO ##########################
sample_frames = all_data_to_combined_csv('./data/Sample_Data', f'sample_{WINDOW_SEC}s.csv', window_sec=WINDOW_SEC)
class_frames5 = all_data_to_combined_csv('./data/Class_Data', f'class_{WINDOW_SEC}s.csv', window_sec=WINDOW_SEC)
print(len(sample_frames), 'sample windows,', len(class_frames5), 'class windows')

DEPTHS = [2, 3, 4, 5, 6, 8, 10, 12, 15, None]
CRITERIA = ['gini', 'entropy']

def quiet(fn, *args, **kwargs):
    """Run a helper that prints a classification report without the printout."""
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)

def grid_search_tree(frames):
    rows = []
    for criterion in CRITERIA:
        for depth in DEPTHS:
            _, _, acc = quiet(train_decision_tree, frames, max_depth=depth, criterion=criterion, features=FEATURES)
            rows.append({'criterion': criterion, 'max_depth': depth, 'split_acc': acc})
    grid = pd.DataFrame(rows)
    # highest split accuracy; ties go to the shallower tree
    grid['_depth'] = grid['max_depth'].fillna(1000)
    best = grid.sort_values(['split_acc', '_depth'], ascending=[False, True]).iloc[0]
    return grid.drop(columns='_depth'), {'criterion': best['criterion'],
                                          'max_depth': None if pd.isna(best['max_depth']) else int(best['max_depth'])}

def show_cm(y_true, y_pred, title, fname):
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(f'figures/{fname}', dpi=150); plt.show()
    return cm

print('===== Default tree (max_depth=5, entropy), trained on Sample_Data =====')
dt1_default, _, acc1_default_split = train_decision_tree(sample_frames, features=FEATURES)
print('--- evaluated on Class_Data ---')
acc1_default_class, _ = evaluate(dt1_default, class_frames5)

grid1, best1 = grid_search_tree(sample_frames)
print(grid1.pivot(index='max_depth', columns='criterion', values='split_acc').round(3))
print('best hyperparameters (by Sample_Data split):', best1)

print('===== Best tree, trained on Sample_Data =====')
dt1_best, _, acc1_best_split = train_decision_tree(sample_frames, features=FEATURES, **best1)
print('--- evaluated on Class_Data ---')
acc1_best_class, pred1 = evaluate(dt1_best, class_frames5)
cm1 = show_cm(class_frames5['activity'], pred1, 'DT (Sample_Data -> Class_Data)', 'cm_c1_dt_sample_to_class.png')
viz_tree(dt1_best, class_frames5[FEATURES], list(dt1_best.classes_))
plt.savefig('figures/tree_c1.png', dpi=120); plt.show()

RESULTS['c1'] = {'default': {'split': acc1_default_split, 'test': acc1_default_class},
                 'best_params': best1, 'best': {'split': acc1_best_split, 'test': acc1_best_class},
                 'grid': grid1.to_dict('records')}
'''

CLASS2 = r'''# roughly similar flow as previous section

########################### TODO ##########################
print('===== Default tree (max_depth=5, entropy), trained on Class_Data =====')
dt2_default, _, acc2_default_split = train_decision_tree(class_frames5, features=FEATURES)
print('--- evaluated on Sample_Data ---')
acc2_default_sample, _ = evaluate(dt2_default, sample_frames)

grid2, best2 = grid_search_tree(class_frames5)
print(grid2.pivot(index='max_depth', columns='criterion', values='split_acc').round(3))
print('best hyperparameters (by Class_Data split):', best2)

print('===== Best tree, trained on Class_Data =====')
dt2_best, cm2_split, acc2_best_split = train_decision_tree(class_frames5, features=FEATURES, **best2)
print('--- evaluated on Sample_Data ---')
acc2_best_sample, pred2 = evaluate(dt2_best, sample_frames)
cm2 = show_cm(sample_frames['activity'], pred2, 'DT (Class_Data -> Sample_Data)', 'cm_c2_dt_class_to_sample.png')
viz_tree(dt2_best, class_frames5[FEATURES], list(dt2_best.classes_))
plt.savefig('figures/tree_c2.png', dpi=120); plt.show()

RESULTS['c2'] = {'default': {'split': acc2_default_split, 'test': acc2_default_sample},
                 'best_params': best2, 'best': {'split': acc2_best_split, 'test': acc2_best_sample},
                 'grid': grid2.to_dict('records')}
'''

CLASS3 = r'''# Hints, step by step:
# Write a code for evaluation and training the new classifier
# Train the new classifier
# Display the confusion matrix
# Visualize the model if it makes sense (a tree, or feature importances for a forest)
# Evaluate the model on class dataset

########################### TODO ##########################
def train_model(frames, model, features=None, verbose=True):
    """train_decision_tree for any sklearn classifier: same 70/30 split (random_state=42)."""
    if features is None:
        features = feature_columns(frames)
    X_train, X_test, y_train, y_test = train_test_split(frames[list(features)], frames['activity'],
                                                        test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    if verbose:
        print(classification_report(y_test, pred, zero_division=0))
        print("Accuracy on test set:", acc)
    return model, confusion_matrix(y_test, pred, labels=model.classes_), acc

# Extra Trees = a random forest that also draws the split thresholds at random, which averages out
# more of the person-to-person noise (it beat the random forest with recordings held out, Model Exploration).
ET_GRID = [{'n_estimators': n, 'max_features': mf, 'min_samples_leaf': leaf, 'max_depth': d}
           for n in [200, 500] for mf in ['sqrt', 0.5] for leaf in [1, 2, 5] for d in [20, None]]

def make_et(p):
    return ExtraTreesClassifier(random_state=42, n_jobs=-1, **p)

def grid_search_et(frames):
    rows = []
    for p in ET_GRID:
        _, _, acc = train_model(frames, make_et(p), FEATURES, verbose=False)
        rows.append({**p, 'split_acc': acc})
    grid = pd.DataFrame(rows)
    # highest split accuracy; ties go to fewer trees / larger leaves
    best = grid.sort_values(['split_acc', 'n_estimators', 'min_samples_leaf'], ascending=[False, True, False]).iloc[0]
    params = {'n_estimators': int(best['n_estimators']),
              'max_features': best['max_features'] if isinstance(best['max_features'], str) else float(best['max_features']),
              'min_samples_leaf': int(best['min_samples_leaf']),
              'max_depth': None if pd.isna(best['max_depth']) else int(best['max_depth'])}
    return grid, params

def plot_importances(model, title, fname):
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    imp.plot.barh(figsize=(7, 4), title=title)
    plt.xlabel('mean decrease in impurity'); plt.tight_layout(); plt.savefig(f'figures/{fname}', dpi=150); plt.show()
    return imp

grid3, best3 = grid_search_et(sample_frames)
print(grid3.round(3).to_string())
print('best hyperparameters (by Sample_Data split):', best3)

print('===== Best extra trees, trained on Sample_Data =====')
et3_best, _, acc3_best_split = train_model(sample_frames, make_et(best3), FEATURES)
print('--- evaluated on Class_Data ---')
acc3_best_class, pred3 = evaluate(et3_best, class_frames5)
cm3 = show_cm(class_frames5['activity'], pred3, 'Extra Trees (Sample_Data -> Class_Data)', 'cm_c3_et_sample_to_class.png')
imp3 = plot_importances(et3_best, 'Extra Trees feature importance (trained on Sample_Data)', 'imp_c3.png')

f1_dt1 = f1_score(class_frames5['activity'], pred1, labels=CLASSES, average=None, zero_division=0)
f1_rf3 = f1_score(class_frames5['activity'], pred3, labels=CLASSES, average=None, zero_division=0)
print(pd.DataFrame({'DT F1': f1_dt1, 'ET F1': f1_rf3, 'gain': f1_rf3 - f1_dt1}, index=CLASSES).round(3))

RESULTS['c3'] = {'best_params': best3, 'best': {'split': acc3_best_split, 'test': acc3_best_class},
                 'f1_dt': dict(zip(CLASSES, f1_dt1)), 'f1_rf': dict(zip(CLASSES, f1_rf3))}
'''

CLASS4 = r'''# roughly similar flow as previous section

########################### TODO ##########################
grid4, best4 = grid_search_et(class_frames5)
print(grid4.round(3).to_string())
print('best hyperparameters (by Class_Data split):', best4)

print('===== Best extra trees, trained on Class_Data =====')
et4_best, _, acc4_best_split = train_model(class_frames5, make_et(best4), FEATURES)
print('--- evaluated on Sample_Data ---')
acc4_best_sample, pred4 = evaluate(et4_best, sample_frames)
cm4 = show_cm(sample_frames['activity'], pred4, 'Extra Trees (Class_Data -> Sample_Data)', 'cm_c4_et_class_to_sample.png')
imp4 = plot_importances(et4_best, 'Extra Trees feature importance (trained on Class_Data)', 'imp_c4.png')

f1_dt2 = f1_score(sample_frames['activity'], pred2, labels=CLASSES, average=None, zero_division=0)
f1_rf4 = f1_score(sample_frames['activity'], pred4, labels=CLASSES, average=None, zero_division=0)
print(pd.DataFrame({'DT F1': f1_dt2, 'ET F1': f1_rf4, 'gain': f1_rf4 - f1_dt2}, index=CLASSES).round(3))

RESULTS['c4'] = {'best_params': best4, 'best': {'split': acc4_best_split, 'test': acc4_best_sample},
                 'f1_dt': dict(zip(CLASSES, f1_dt2)), 'f1_rf': dict(zip(CLASSES, f1_rf4)),
                 'importance': imp4.to_dict()}
'''

MODEL_EXPLORATION = r'''#### Model exploration (exploration report §2)
# Same FEATURES, same 5 s windows. Three numbers per model:
#   split      : the notebook's 70/30 window split of Class_Data (windows of one recording land on both sides)
#   grouped CV : 5-fold CV on Class_Data with whole recordings held out (what a new user would see)
#   C->S, S->C : train on one corpus, test on the other
MODELS = {
    'Decision Tree (entropy, depth 5)': lambda: DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42),
    f'Decision Tree ({best2["criterion"]}, depth {best2["max_depth"]})': lambda: DecisionTreeClassifier(random_state=42, **best2),
    'Random Forest (300 trees)': lambda: RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    f'Extra Trees {best4}': lambda: make_et(best4),
    'HistGradientBoosting (lr 0.05, 300 it)': lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, random_state=42),
    'Gradient Boosting (100, depth 3)': lambda: GradientBoostingClassifier(random_state=42),
    'kNN (k=15, scaled)': lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
    'Logistic Regression (C=1, scaled)': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=5000)),
    'SVM (RBF, C=3, scaled)': lambda: make_pipeline(StandardScaler(), SVC(C=3, random_state=42)),
}
model_rows = []
for name, make in MODELS.items():
    _, _, split_acc = train_model(class_frames5, make(), FEATURES, verbose=False)
    cv_acc, cv_pred = grouped_cv(class_frames5, FEATURES, make)
    c2s = accuracy_score(sample_frames['activity'], make().fit(class_frames5[FEATURES], class_frames5['activity']).predict(sample_frames[FEATURES]))
    s2c = accuracy_score(class_frames5['activity'], make().fit(sample_frames[FEATURES], sample_frames['activity']).predict(class_frames5[FEATURES]))
    f1 = f1_score(class_frames5['activity'], cv_pred, labels=CLASSES, average=None, zero_division=0)
    model_rows.append({'model': name, 'split': split_acc, 'grouped_cv': cv_acc, 'C->S': c2s, 'S->C': s2c,
                       **{f'f1_{c}': v for c, v in zip(CLASSES, f1)}})
model_table = pd.DataFrame(model_rows)
print(model_table.round(3).to_string())
RESULTS['models'] = model_rows
'''

PREDICT_REC = r'''def predict_recordings(model, frames, features=None):
    # By default, use the same columns the model was trained on
    if features is None:
        features = getattr(model, 'feature_names_in_', feature_columns(frames))

    ######## TODO ###############
    # Sample_Data reuses file names (026.csv ...) in every activity folder, so a recording is
    # identified by (activity folder, file name), not by the file name alone.
    pred = model.predict(frames[list(features)])
    proba = model.predict_proba(frames[list(features)]) if hasattr(model, 'predict_proba') else None
    keys = ['activity', 'recording'] if 'activity' in frames.columns else ['recording']
    rows = []
    for key, idx in frames.groupby(keys).indices.items():
        votes = pd.Series(pred[idx]).value_counts()
        tied = list(votes.index[votes == votes.max()])
        if len(tied) > 1 and proba is not None:
            # majority vote; a tie goes to the tied label with the highest summed probability
            summed = dict(zip(model.classes_, proba[idx].sum(axis=0)))
            label = max(tied, key=lambda c: summed[c])
        else:
            label = tied[0]
        key = key if isinstance(key, tuple) else (key,)
        rows.append({'recording': key[-1], 'true': frames['activity'].iloc[idx[0]] if 'activity' in frames.columns else None,
                     'predicted': label, 'windows': len(idx)})
    recordings = pd.DataFrame(rows)
    #############################

    return recordings
'''

CLASS5 = r'''# Hints, step by step:
# Call predict_recordings for each model and test set
# Recording-level accuracy = (recordings['true'] == recordings['predicted']).mean()
# Display a confusion matrix of the recording-level predictions

########################### TODO ##########################
def confidence_vote(model, frames, features):
    """Alternative aggregation: sum predict_proba over the windows of a recording, take the argmax."""
    proba = pd.DataFrame(model.predict_proba(frames[features]), columns=model.classes_, index=frames.index)
    summed = proba.groupby([frames['activity'], frames['recording']]).sum()
    return (summed.idxmax(axis=1) == summed.index.get_level_values(0)).mean()

runs = [('Decision Tree (best)', dt2_best, 'Class_Data -> Sample_Data', sample_frames, acc2_best_sample),
        ('Extra Trees (best)', et4_best, 'Class_Data -> Sample_Data', sample_frames, acc4_best_sample),
        ('Decision Tree (best)', dt1_best, 'Sample_Data -> Class_Data', class_frames5, acc1_best_class),
        ('Extra Trees (best)', et3_best, 'Sample_Data -> Class_Data', class_frames5, acc3_best_class)]
rec_rows = []
for name, model, direction, test, win_acc in runs:
    recs = predict_recordings(model, test, FEATURES)
    rec_acc = (recs['true'] == recs['predicted']).mean()
    per_class = recs.assign(ok=recs['true'] == recs['predicted']).groupby('true')['ok'].mean()
    win_per_class = pd.Series(model.predict(test[FEATURES]) == test['activity'].values).groupby(test['activity'].values).mean()
    rec_rows.append({'model': name, 'direction': direction, 'window_acc': win_acc, 'recording_acc': rec_acc,
                     'confidence_vote_acc': confidence_vote(model, test, FEATURES), 'recordings': len(recs),
                     'per_class_window': win_per_class.to_dict(), 'per_class_recording': per_class.to_dict()})
    print(f'{name:22s} {direction}: window {win_acc:.3f}  recording {rec_acc:.3f}  ({len(recs)} recordings)')
    print(pd.DataFrame({'window acc': win_per_class, 'recording acc': per_class}).round(3).T.to_string(), '\n')
    show_cm(recs['true'], recs['predicted'], f'{name}, {direction} (recordings)',
            f"cm_rec_{name.split()[0].lower()}_{direction.split()[0].lower()}.png")
RESULTS['recordings'] = rec_rows
'''

FAILURE = r'''# Hints, step by step:
# Find misclassified windows: frames[pred != frames['activity']]
# Pick one recording, reload its CSV, apply calc_magnitude and remove_noise, and plot filtered_accel_mag
# Compare: frames.groupby('activity')[FEATURES].median() vs the feature values of the failing window

########################### TODO ##########################
# Final pipeline (report §3): 5 s windows, FEATURES, random forest with the Class_Data-tuned hyperparameters,
# majority vote per recording. Its confusion matrix comes from 5-fold CV on Class_Data with whole recordings
# held out, so every window is predicted by a model that never saw its recording.
final_make = lambda: make_et(best4)
cv_acc, oof = grouped_cv(class_frames5, FEATURES, final_make)
frames = class_frames5.assign(pred=oof)
cm = confusion_matrix(frames['activity'], frames['pred'], labels=CLASSES)
cm_df = pd.DataFrame(cm, index=[f'true {c}' for c in CLASSES], columns=[f'pred {c}' for c in CLASSES])
print('window accuracy (recordings held out):', round(cv_acc, 3))
print(cm_df.to_string())
show_cm(frames['activity'], frames['pred'], 'Final ET, Class_Data grouped 5-fold CV (windows)', 'cm_final_windows.png')

rec = (frames.groupby(['activity', 'recording'])['pred'].agg(lambda s: s.value_counts().idxmax())
       .rename('predicted').reset_index().rename(columns={'activity': 'true'}))
rec_cv_acc = (rec['true'] == rec['predicted']).mean()
print('recording accuracy (majority vote, recordings held out):', round(rec_cv_acc, 3))
show_cm(rec['true'], rec['predicted'], 'Final ET, Class_Data grouped 5-fold CV (recordings)', 'cm_final_recordings.png')

# top-2 failure modes = largest off-diagonal cells
off = cm_df.copy().values.astype(float); np.fill_diagonal(off, 0)
order = np.dstack(np.unravel_index(np.argsort(-off.ravel()), off.shape))[0][:6]
top = [(CLASSES[i], CLASSES[j], int(cm[i, j]), cm[i, j] / cm[i].sum()) for i, j in order]
for t in top:
    print(f'{t[0]:>10s} predicted as {t[1]:<10s}: {t[2]} windows ({t[3]:.0%} of {t[0]})')

medians = frames.groupby('activity')[FEATURES].median()
print(medians.round(3).to_string())

def show_failure(true_c, pred_c, k):
    # recording with the most windows of this confusion (at least 5 windows, mostly wrong)
    bad = frames[(frames['activity'] == true_c)]
    stats = bad.groupby('recording')['pred'].agg(n='size', hits=lambda s: (s == pred_c).sum())
    stats = stats[stats['n'] >= 5]
    stats['frac'] = stats['hits'] / stats['n']
    rec_name = stats.sort_values(['frac', 'n'], ascending=False).index[0]
    win = bad[(bad['recording'] == rec_name) & (bad['pred'] == pred_c)].iloc[0]
    print(f'\nFailure {k}: {true_c} -> {pred_c}, recording {true_c}/{rec_name}: '
          f'{int(stats.loc[rec_name, "hits"])}/{int(stats.loc[rec_name, "n"])} windows predicted {pred_c}')

    raw = pd.read_csv(f'./data/Class_Data/{true_c}/{rec_name}', parse_dates=['time'])[['time', 'x', 'y', 'z']]
    dt = raw['time'].diff().dt.total_seconds()
    fs = 1 / dt[dt > 0].median()
    raw = remove_noise(calc_magnitude(raw), fs)
    t = (raw['time'] - raw['time'].iloc[0]).dt.total_seconds()

    # a typical correctly classified recording of the true class, for comparison
    good = frames[(frames['activity'] == true_c)].groupby('recording')['pred'].agg(n='size', ok=lambda s: (s == true_c).mean())
    good_name = good[good['n'] >= 5].sort_values('ok', ascending=False).index[0]
    ref = pd.read_csv(f'./data/Class_Data/{true_c}/{good_name}', parse_dates=['time'])[['time', 'x', 'y', 'z']]
    dtr = ref['time'].diff().dt.total_seconds()
    ref = remove_noise(calc_magnitude(ref), 1 / dtr[dtr > 0].median())
    tr = (ref['time'] - ref['time'].iloc[0]).dt.total_seconds()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.5, 1]})
    ax = axes[0]
    ax.plot(t, raw['filtered_accel_mag'], lw=0.7, label=f'failing: {true_c}/{rec_name[:30]}')
    ax.plot(tr, ref['filtered_accel_mag'] - 0, lw=0.7, alpha=0.6, label=f'correct {true_c}: {good_name[:30]}')
    # shade the windows of the failing recording that went wrong
    ws = frames[(frames['activity'] == true_c) & (frames['recording'] == rec_name)]
    rec_windows = pd.Series(raw['time'].values).dt.floor(f'{WINDOW_SEC}s')
    starts = sorted(rec_windows.unique())
    keep = [s for s in starts if (rec_windows == s).sum() >= 0.8 * WINDOW_SEC * fs]
    for s, p in zip(keep, ws['pred']):
        x0 = (s - raw['time'].iloc[0]).total_seconds()
        ax.axvspan(max(x0, 0), x0 + WINDOW_SEC, color='tab:red' if p != true_c else 'tab:green', alpha=0.08)
    # y range from the 1st-99th percentile of both traces so a start-up spike doesn't flatten the plot
    lim = 1.3 * max(np.percentile(np.abs(raw['filtered_accel_mag']), 99), np.percentile(np.abs(ref['filtered_accel_mag']), 99))
    ax.set_ylim(-lim, lim)
    ax.set_xlim(0, min(60, t.iloc[-1])); ax.set_xlabel('time (s)'); ax.set_ylabel('filtered_accel_mag (m/s²)')
    ax.set_title(f'{true_c} classified as {pred_c} (red = wrong window, green = right)'); ax.legend(fontsize=8, loc='upper right')

    ax = axes[1]
    comp = pd.DataFrame({'failing window': win[FEATURES].astype(float),
                         f'median {true_c} (true)': medians.loc[true_c],
                         f'median {pred_c} (predicted)': medians.loc[pred_c]})
    scale = comp.abs().max(axis=1).replace(0, 1)
    (comp.div(scale, axis=0)).plot.barh(ax=ax, width=0.8)
    ax.set_title('feature values, scaled by row max'); ax.legend(fontsize=8); ax.invert_yaxis()
    plt.tight_layout(); plt.savefig(f'figures/failure_{k}.png', dpi=150); plt.show()
    print(comp.round(3).to_string())
    return {'true': true_c, 'pred': pred_c, 'recording': rec_name, 'reference': good_name,
            'hits': int(stats.loc[rec_name, 'hits']), 'n': int(stats.loc[rec_name, 'n']),
            'values': comp.round(4).to_dict()}

failures = [show_failure(top[0][0], top[0][1], 1), show_failure(top[1][0], top[1][1], 2)]

RESULTS['final'] = {'params': best4, 'features': FEATURES, 'window_sec': WINDOW_SEC,
                    'grouped_cv_window_acc': cv_acc, 'grouped_cv_recording_acc': rec_cv_acc,
                    'cm': cm.tolist(), 'classes': CLASSES, 'top_confusions': top,
                    'medians': medians.round(4).to_dict(), 'failures': failures,
                    'rec_cm': confusion_matrix(rec['true'], rec['predicted'], labels=CLASSES).tolist()}
with open('results.json', 'w') as f:
    json.dump(RESULTS, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print('saved results.json')
'''
