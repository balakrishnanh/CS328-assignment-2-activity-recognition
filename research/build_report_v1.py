"""Build exploration_report.pdf and writeup.pdf from results.json and figures/ (run after the notebook)."""
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether

R = json.load(open('results.json'))
CLASSES = R['final']['classes']
ft = {r['set']: r for r in R['feature_table']}
base, fin = ft['baseline (7)'], ft['FINAL (8)']
p = lambda v: f'{100 * v:.1f}%'
pts = lambda v: f'{100 * v:+.1f}'

ss = getSampleStyleSheet()
body = ParagraphStyle('b', parent=ss['BodyText'], fontSize=9, leading=11.5, spaceAfter=4)
small = ParagraphStyle('s', parent=body, fontSize=7.5, leading=9)
h1 = ParagraphStyle('h1', parent=ss['Heading1'], fontSize=15, spaceAfter=4)
h2 = ParagraphStyle('h2', parent=ss['Heading2'], fontSize=11.5, spaceBefore=8, spaceAfter=3)
cap = ParagraphStyle('c', parent=body, fontSize=7.5, textColor=colors.HexColor('#444444'))

def P(t, s=body): return Paragraph(t, s)

def T(rows, widths, header=True, font=7.5):
    data = [[P(str(c), ParagraphStyle('t', parent=small, fontSize=font, leading=font + 1.5)) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#999999')),
          ('VALIGN', (0, 0), (-1, -1), 'TOP'),
          ('LEFTPADDING', (0, 0), (-1, -1), 3), ('RIGHTPADDING', (0, 0), (-1, -1), 3),
          ('TOPPADDING', (0, 0), (-1, -1), 1.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5)]
    if header:
        st.append(('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8e8e8')))
    t.setStyle(TableStyle(st))
    return t

def img(path, w):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    return Image(path, width=w, height=w * ih / iw)

# ------------------------------------------------------------------ §1
s = []
s.append(P('CS328 A2 — Activity Classification: Exploration Report', h1))
s.append(P('Data: 615 Class_Data recordings (Sensor Logger phones, gravity removed) and 35 Sample_Data recordings '
           '(wrist device, 5 people). All numbers come from the submitted notebook (<i>results.json</i>). '
           '"Split" is the notebook\'s 70/30 random window split. Because it puts windows of one recording on both sides, '
           'we also report <b>grouped CV</b>: 5-fold CV on Class_Data with whole recordings held out, which is what a new user '
           'would see. We chose features and models on grouped CV, not on the leaky split.', body))

s.append(P('§1 Feature exploration', h2))
s.append(P(f'<b>Window length</b> (baseline features, depth-5 entropy tree): 2 s {p(R["window_baseline"][0]["split_acc"])}, '
           f'3 s {p(R["window_baseline"][1]["split_acc"])}, 5 s {p(R["window_baseline"][2]["split_acc"])}, '
           f'10 s {p(R["window_baseline"][3]["split_acc"])} on the split. Grouped CV is flat (48-50%) at every length, '
           'so window length is not the bottleneck. We chose <b>5 s, non-overlapping</b>: about 9 steps per window (enough for '
           'autocorrelation and 0.2 Hz FFT resolution), 612 of 615 recordings keep at least one window (10 s keeps 603), and a '
           'typical 40 s recording gets 5-10 votes.', body))
s.append(P('Each row adds one feature to the 7 baseline statistics (avg, max, med, min, q25, q75, std of the filtered magnitude). '
           '"Change, split" is the depth-5 tree on the notebook split; "Change, grouped CV" is a random forest (200 trees) with recordings '
           f'held out; "drop from final" is the grouped-CV change when that feature is removed from the final 8 '
           f'(final = {p(fin["rf_grouped_cv"])}). The class is the one whose F1 rose the most.', body))

rationale = {
    'min': ('(baseline) static vs gait', 'Lowest point of the detrended magnitude: deep dips on every step, near 0 at rest. Top RF importance (0.23).', 'kept'),
    'axis_std': ('upstairs vs walk (F1 0→0.28)', 'sqrt(trace of the 3x3 covariance of x,y,z): total 3-D motion, the same however the phone is rotated, unlike std_x/y/z.', 'kept'),
    'jerk_raw': ('downstairs (F1 0→0.42)', 'RMS sample-to-sample change of x,y,z (m/s³): heel-strike impacts going down are sharp.', 'kept'),
    'hf_frac': ('downstairs; standing vs sitting/lying', 'Share of unfiltered power above 5 Hz. Standing sway is slow (0.31); sitting/lying are flat so sensor noise dominates (0.50).', 'kept'),
    'peaks_per_sec': ('jogging vs walk; downstairs', 'Step peaks ≥0.25 s apart. Jogging cadence (2.4/s) > walking (1.8/s).', 'kept'),
    'step_period': ('stairs (downstairs F1 0→0.46, upstairs 0→0.18)', 'Lag of the autocorrelation peak in 0.25-1.5 s. Stairs are slower and more regular per step than walking.', 'kept'),
    'spec_entropy': ('downstairs (0→0.50), lying (0.09→0.28)', 'Flatness of the 0.3-5 Hz spectrum: gait is peaky (one cadence), rest is flat noise.', 'kept'),
    'low_band_frac': ('downstairs (0→0.39); standing', 'Share of 0.3-1 Hz power: body sway and slow posture shifts vs step energy at 1.5-3 Hz.', 'kept'),
    'rms': ('none', 'Redundant with std and min (the window mean is already in avg).', 'dropped'),
    'zcr': ('downstairs, jogging', 'Rhythm proxy; redundant once peaks_per_sec and step_period are in.', 'dropped'),
    'acf_peak': ('downstairs (0→0.53)', 'Regularity of the gait; strong alone but overlaps with spec_entropy.', 'dropped'),
    'dom_freq': ('lying, downstairs', 'Cadence from the FFT peak; harmonics make it jump between step and stride rate.', 'dropped'),
    'std_x': ('none', 'Depends on how the phone sits in the pocket; does not transfer to the wrist.', 'dropped'),
    'std_y': ('downstairs (+0.15)', 'Same orientation problem.', 'dropped'),
    'std_z': ('downstairs (+0.22)', 'Same orientation problem; replaced by axis_std (a std_z set transferred 1-2 points worse between devices in research/exp8.py).', 'dropped'),
    'pca1': ('none', 'Share of motion along the main axis; no gain.', 'dropped'),
}
rows = [['Feature', 'Class it helps discriminate', 'Change, split (DT)', 'Change, grouped CV (RF)', 'Drop from final', 'Rationale', 'Final?']]
for f, (cls, why, keep) in rationale.items():
    r = ft['baseline (7)'] if f == 'min' else ft[f'baseline + {f}']
    drop = ft.get(f'final - {f}')
    rows.append([f'<b>{f}</b>' if keep == 'kept' else f, cls,
                 '—' if f == 'min' else pts(r['dt_split'] - base['dt_split']),
                 '—' if f == 'min' else pts(r['rf_grouped_cv'] - base['rf_grouped_cv']),
                 pts(drop['rf_grouped_cv'] - fin['rf_grouped_cv']) if drop else '—', why, keep])
rows.append(['<b>final 8</b>', 'all', pts(fin['dt_split'] - base['dt_split']), pts(fin['rf_grouped_cv'] - base['rf_grouped_cv']), '', 'min + 7 above', ''])
rows.append(['baseline + all 15', 'all', pts(ft['baseline + all candidates']['dt_split'] - base['dt_split']),
             pts(ft['baseline + all candidates']['rf_grouped_cv'] - base['rf_grouped_cv']), '', '22 features: only +1.0 over our 8', ''])
s.append(T(rows, [0.85 * inch, 1.2 * inch, 0.55 * inch, 0.6 * inch, 0.55 * inch, 2.9 * inch, 0.55 * inch]))
s.append(Spacer(1, 3))
s.append(P('How we chose: greedy forward selection with recordings held out (research/exp7.py) picked min, step_period, std_z, '
           'peaks_per_sec, spec_entropy, hf_frac, low_band_frac; we swapped std_z for the rotation-invariant axis_std and added '
           'jerk_raw, because per-axis features depend on phone placement and transferred worse to the wrist data. '
           'Honest note: the leave-one-out column shows jerk_raw, low_band_frac and axis_std each change grouped CV by '
           'less than ±0.5 points, which is inside fold noise, so a 5-6 feature set would perform about the same. The features that '
           'clearly carry weight are min, step_period, peaks_per_sec and spec_entropy (each −1 to −2 points when removed).', body))

# ------------------------------------------------------------------ §2
s.append(P('§2 Model exploration', h2))
s.append(P('Same 8 features and 5 s windows for every model. Hyperparameters for the tree and forest come from the notebook grid '
           'searches (Classification 2 and 4, chosen on the Class_Data split).', body))
rows = [['Model', 'Key hyperparameter(s)', 'Split', 'Grouped CV', 'Class→Sample', 'Sample→Class', 'F1 upstairs / lying', 'Notes']]
notes = {
    0: 'Baseline, interpretable. Never predicts upstairs.',
    1: 'Grid winner on the split, but worse on grouped CV: deeper tree memorises recordings.',
    2: 'Best on the split and Class→Sample; used in the final pipeline.',
    3: 'Best grouped CV; extra randomness regularises.',
    4: 'Ties RF on the split; best lying F1.',
    5: 'Needs scaling; weakest Sample→Class.',
    6: 'Linear boundaries are enough for intensity, not for the static classes.',
    7: 'Close to the forests; smooth boundaries.',
}
hp = ['max_depth=5, entropy', f'max_depth={R["c2"]["best_params"]["max_depth"]}, {R["c2"]["best_params"]["criterion"]}',
      f'n_estimators={R["c4"]["best_params"]["n_estimators"]}, max_depth={R["c4"]["best_params"]["max_depth"]}, min_samples_leaf=1',
      'n_estimators=300, min_samples_leaf=2', 'n_estimators=100, max_depth=3, lr=0.1', 'k=15, StandardScaler',
      'C=1, StandardScaler', 'RBF, C=3, StandardScaler']
names = ['Decision Tree', 'Decision Tree (tuned)', 'Random Forest', 'Extra Trees', 'Gradient Boosting', 'kNN', 'Logistic Regression', 'SVM']
for i, m in enumerate(R['models']):
    rows.append([names[i], hp[i], p(m['split']), p(m['grouped_cv']), p(m['C->S']), p(m['S->C']),
                 f"{m['f1_upstairs']:.2f} / {m['f1_lying']:.2f}", notes[i]])
s.append(T(rows, [1.0 * inch, 1.35 * inch, 0.45 * inch, 0.55 * inch, 0.6 * inch, 0.6 * inch, 0.65 * inch, 2.0 * inch]))
c4 = R['c4']
s.append(Spacer(1, 3))
s.append(P(f'<b>Which won and why.</b> The tree ensembles (random forest, extra trees, gradient boosting) and the RBF SVM form a top group '
           f'at 57-59% grouped CV, about 7 points above the best single tree (50.7%). We picked the <b>random forest</b>: best on the '
           f'notebook split ({p(R["models"][2]["split"])}) and on Class→Sample ({p(R["models"][2]["C->S"])}), within a point of extra trees '
           f'on grouped CV, and it gives feature importances. A single tree must spend each split on one threshold, so classes that '
           f'need a combination of weak cues (upstairs: slower rhythm and lower jerk; lying: noise level and low-band share) get '
           f'few or no leaves. Averaging many decorrelated trees covers those regions. <b>The gain was not equal across classes.</b> '
           f'Class→Sample, the forest raised upstairs F1 {c4["f1_dt"]["upstairs"]:.2f}→{c4["f1_rf"]["upstairs"]:.2f} and lying '
           f'{c4["f1_dt"]["lying"]:.2f}→{c4["f1_rf"]["lying"]:.2f}, walk +0.10 and downstairs +0.08, while jogging was already '
           f'{c4["f1_dt"]["jogging"]:.2f} and stayed there. The tuned-tree row also shows the cost of tuning on the leaky split: '
           f'depth 8 wins the split (+4 points) but loses on grouped CV (−1.1) and on Sample_Data (−4.6).', body))

# ------------------------------------------------------------------ §3
fp = R['final']
s.append(P('§3 Universal pipeline', h2))
s.append(T([
    ['Step', 'Choice'],
    ['Preprocessing', 'Sampling rate per file = 1 / median timestamp step (50-101 Hz in the data); magnitude of (x,y,z), detrended by the '
                      'recording mean (calc_magnitude); 2nd-order Butterworth low-pass at 5 Hz, zero-phase (remove_noise).'],
    ['Windows', '5 s, non-overlapping, clock-aligned via DataFrame.resample; windows with &lt;80% of the expected samples are dropped.'],
    ['Features (8)', 'min, axis_std, jerk_raw, hf_frac, peaks_per_sec, step_period, spec_entropy, low_band_frac'],
    ['Classifier', f'RandomForestClassifier(n_estimators={fp["params"]["n_estimators"]}, max_depth={fp["params"]["max_depth"]}, '
                   f'min_samples_leaf={fp["params"]["min_samples_leaf"]}, random_state=42), trained on Class_Data'],
    ['Aggregation', 'Majority vote of the window labels per recording (recording = activity folder + file name, since Sample_Data reuses '
                    '026.csv…030.csv in every folder); ties broken by the summed predict_proba.'],
    ['Performance', f'Grouped CV on Class_Data: {p(fp["grouped_cv_window_acc"])} of windows, {p(fp["grouped_cv_recording_acc"])} of '
                    f'recordings. Class→Sample: {p(c4["best"]["test"])} windows, {p(R["recordings"][1]["recording_acc"])} recordings. '
                    f'Notebook split: {p(c4["best"]["split"])}.'],
], [1.1 * inch, 6.1 * inch]))
s.append(Spacer(1, 3))
s.append(P('<b>How we decided.</b> We locked one axis at a time. Window length first, with baseline features, because nothing else '
           'depended on it and the grouped CV showed it barely mattered. Features next, judged with a random forest under grouped CV, '
           'because the 30% random split leaks windows of the same recording into test and rewards memorising recordings (it '
           'consistently preferred deeper trees that then did worse on new recordings and on Sample_Data). We kept features that '
           'are rotation-invariant or computed from the magnitude, so the pipeline does not depend on how the phone sits in the '
           'pocket and still transfers to a wrist device. Model last: the random forest was best or tied-best on every protocol. '
           'For aggregation we compared majority vote with confidence weighting (summed probabilities): it helped the tree but '
           'not the forest (62.9% vs 65.7% Class→Sample), so we kept the majority vote.', body))

# ------------------------------------------------------------------ §4
s.append(P('§4 Confusion matrix + per-class analysis', h2))
cm = fp['cm']   # rows = true, cols = predicted
s.append(P(f'Final pipeline, Class_Data, grouped 5-fold CV, window level ({sum(map(sum, cm))} windows, accuracy '
           f'{p(fp["grouped_cv_window_acc"])}). Rows = predicted, columns = actual, as in the template.', body))
rows = [['Predicted ↓ / Actual →'] + CLASSES]
for j, c in enumerate(CLASSES):
    rows.append([f'<b>{c}</b>'] + [f'<b>{cm[i][j]}</b>' if i == j else str(cm[i][j]) for i in range(len(CLASSES))])
rows.append(['recall'] + [f'{cm[i][i] / sum(cm[i]):.0%}' for i in range(len(CLASSES))])
s.append(T(rows, [1.3 * inch] + [0.8 * inch] * 7))
rc = fp['rec_cm']
s.append(P(f'Recording level (majority vote, same CV): {p(fp["grouped_cv_recording_acc"])}; per-class recall '
           + ', '.join(f'{c} {rc[i][i] / max(1, sum(rc[i])):.0%}' for i, c in enumerate(CLASSES)) + '.', small))
s.append(Spacer(1, 3))
an = [
    ['Cell (actual → predicted)', 'Count', 'Cause', 'Axis'],
    ['lying → sitting', cm[6][5], 'With gravity removed, a still phone reads ≈0 whether the body is horizontal or upright. Medians are almost identical (axis_std 0.086 vs 0.070, hf_frac 0.50 vs 0.51, step_period 0.37 vs 0.36).', 'Feature'],
    ['upstairs → walk', cm[1][0], 'Same intensity and cadence: median min −2.37 vs −2.38, axis_std 3.72 vs 3.78, jerk_raw 74.0 vs 73.8. Only step_period differs (0.74 vs 1.03 s), and it flips between step and stride period.', 'Feature'],
    ['standing → sitting', cm[4][5], 'Standing has more sway (axis_std 0.22 vs 0.07), but a person standing very still overlaps sitting; the threshold depends on the person and where the phone is.', 'Feature'],
    ['sitting → standing', cm[5][4], 'The mirror image: fidgeting or holding the phone while sitting adds sway-like motion.', 'Feature'],
    ['downstairs → walk', cm[2][0], 'A slow, careful descent loses the sharp landing impacts (jerk_raw) that set downstairs apart.', 'Feature'],
    ['sitting → lying', cm[5][6], 'Same as lying→sitting: no orientation signal.', 'Feature'],
    ['walk → downstairs / upstairs', f'{cm[0][2]} / {cm[0][1]}', 'Brisk or heavy walkers produce stair-like jerk and amplitude. The single tree mislabels more of these than the forest (walk F1 0.60 vs 0.66), so part of this is model side for the tree.', 'Feature (model for DT)'],
    ['lying → standing', cm[6][4], 'Likely recordings where the phone is held in the hand while lying: hand tremor looks like standing sway (lying axis_std spread is wide).', 'Feature'],
    ['upstairs → downstairs; downstairs → upstairs', f'{cm[1][2]} / {cm[2][1]}', 'The magnitude has no sign of vertical motion, so ascent and descent look alike apart from impact sharpness. The forest recovers some upstairs the tree missed (F1 0.24→0.27), so a small part is model side.', 'Feature (partly model)'],
    ['jogging → downstairs', cm[3][2], 'Slow jogs with the phone in a loose pocket overlap fast descents in amplitude; medians are far apart (axis_std 15.0 vs 4.3), so this is the tail of the distribution.', 'Feature'],
]
s.append(T(an, [1.45 * inch, 0.55 * inch, 4.4 * inch, 0.8 * inch]))
s.append(P('Overall: almost all of the error is on the <b>feature side</b>. The best model in every family stops at about 58%, and the '
           'per-class medians of the confused pairs are nearly equal, so no classifier can separate them from these features. '
           'The one clear model-side effect is the single tree vs the forest on upstairs and lying (§2).', body))

# ------------------------------------------------------------------ §5
s.append(P('§5 Top failure modes + edge cases', h2))
f1, f2 = fp['failures']
def vals(f, key): return f['values'][key]
v1 = f1['values']; v2 = f2['values']
k1t, k1p = f'median {f1["true"]} (true)', f'median {f1["pred"]} (predicted)'
k2t, k2p = f'median {f2["true"]} (true)', f'median {f2["pred"]} (predicted)'
s.append(P(f'<b>Failure 1: lying classified as sitting</b> ({cm[6][5]} windows, {cm[6][5] / sum(cm[6]):.0%} of all lying windows; only '
           f'{cm[6][6] / sum(cm[6]):.0%} of lying is correct). Example: <i>lying/{f1["recording"]}</i>, {f1["hits"]}/{f1["n"]} windows → sitting.', body))
s.append(img('figures/failure_1.png', 7.2 * inch))
s.append(P(f'Left: the failing trace (blue) vs a correctly classified lying recording; red bands are windows predicted wrong. Right: features '
           f'of one failing window vs the class medians (scaled by row max). The window has hf_frac {v1["failing window"]["hf_frac"]:.2f} '
           f'(lying median {v1[k1t]["hf_frac"]:.2f}, sitting {v1[k1p]["hf_frac"]:.2f}) and axis_std {v1["failing window"]["axis_std"]:.3f} '
           f'(lying {v1[k1t]["axis_std"]:.3f}, sitting {v1[k1p]["axis_std"]:.3f}).', cap))
s.append(P('<b>Mechanism:</b> Sensor Logger subtracts gravity, so the one thing that differs between lying and sitting, which way the '
           'phone (and body) points, is not in the data. What is left is sensor noise and breathing, which depend more on the phone '
           'and surface than on posture. <b>Feature problem:</b> the two classes have the same feature distribution. <b>Fix:</b> also '
           'log Sensor Logger\'s gravity sensor (or the raw accelerometer) and add the direction of gravity in phone coordinates (tilt). For a phone '
           'on the torso or in the hand this separates lying from sitting directly; for a thigh pocket (thigh horizontal in both) it '
           'separates standing from both, and lying vs sitting would also need the rotation about the thigh or a second placement.', body))
s.append(P(f'<b>Failure 2: upstairs classified as walk</b> ({cm[1][0]} windows, {cm[1][0] / sum(cm[1]):.0%} of upstairs windows; more '
           f'upstairs windows are called walk than upstairs). Example: <i>upstairs/{f2["recording"]}</i>, {f2["hits"]}/{f2["n"]} windows → walk.', body))
s.append(img('figures/failure_2.png', 7.2 * inch))
s.append(P(f'Failing window: axis_std {v2["failing window"]["axis_std"]:.2f} (upstairs median {v2[k2t]["axis_std"]:.2f}, walk '
           f'{v2[k2p]["axis_std"]:.2f}), jerk_raw {v2["failing window"]["jerk_raw"]:.1f} ({v2[k2t]["jerk_raw"]:.1f} / '
           f'{v2[k2p]["jerk_raw"]:.1f}), step_period {v2["failing window"]["step_period"]:.2f} s ({v2[k2t]["step_period"]:.2f} / '
           f'{v2[k2p]["step_period"]:.2f}). This climber is gentler than typical and the autocorrelation locked onto a long period, '
           'which is closer to walking\'s stride period.', cap))
s.append(P('<b>Mechanism:</b> climbing at a natural pace has the same cadence and overall magnitude as walking; what differs is that '
           'each step lifts the body (more vertical work, a slower push-off), and that lives on the vertical axis, which the '
           'orientation-free magnitude cannot isolate. Differences between people in amplitude are larger than the difference '
           'between the two activities. <b>Feature problem:</b> the class medians of every intensity feature are within 2%. '
           '<b>Fix:</b> project the acceleration onto the gravity direction (gravity sensor) and add vertical-axis features, '
           'e.g. vertical/horizontal energy ratio and skew of vertical acceleration; or add the barometer, since climbing a '
           'flight changes pressure by about 0.3 hPa.', body))
s.append(P('<b>Edge cases seen in the data:</b> recordings from 4.5 s to 27 min (the 27-min sitting file alone gives 323 windows); '
           'a few phones at 50-58 Hz, which is why the sampling rate is estimated per file; one lying file with a 20 s logging gap '
           '(those windows are skipped); a "walk" recording named "activity_1_-_cooking"; and recordings shorter than 5 s, which '
           'produce no window and are not classified (3 of 615).', body))
s.append(P('References: Bao &amp; Intille 2004 (window features + decision trees for accelerometer HAR); Kwapisz, Weiss &amp; Moore 2011, '
           'WISDM (phone-in-pocket: stairs were the hardest classes); Anguita et al. 2013, UCI-HAR (time- and frequency-domain features, '
           'gravity-separated signals).', small))

doc = SimpleDocTemplate('exploration_report.pdf', pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                        topMargin=0.55 * inch, bottomMargin=0.55 * inch, title='CS328 A2 Exploration Report')
doc.build(s)

# ------------------------------------------------------------------ writeup
w = [P('CS328 A2 — Writeup', h1)]
w.append(P('<b>Most important decision.</b> We judged features and models by holding out whole recordings, not by the notebook\'s random '
           'window split. The split leaks windows of the same recording into the test set, so it kept rewarding deeper trees that '
           'memorise recordings (+4 points on the split, −4.6 points on Sample_Data). Choosing on grouped CV is what led us to '
           f'rotation-invariant rhythm and noise features and to the random forest ({p(fp["grouped_cv_window_acc"])} on unseen recordings, '
           f'vs 48.7% for the baseline tree).', body))
w.append(P(f'<b>Biggest surprise in the confusion matrix.</b> Lying is called sitting {cm[6][5] / sum(cm[6]):.0%} of the time and is right '
           f'only {cm[6][6] / sum(cm[6]):.0%} of the time, worse than guessing among the three static classes, and upstairs is predicted '
           'as walk more often than as upstairs. In both cases the confused classes have almost identical median feature values, so the '
           'missing information is in the sensor, not in the model.', body))
w.append(P('<b>Next.</b> Log the gravity vector (or raw accelerometer) and the barometer, and add tilt-angle and vertical-axis features to '
           'separate lying from sitting and stairs from walking.', body))
w.append(P('<b>Contribution statement.</b> [FILL IN: one line, who did what. Example format: "A: data loading and windowing; B: '
           'feature engineering and model comparison; C: failure analysis and report."]', body))
SimpleDocTemplate('writeup.pdf', pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                  topMargin=0.7 * inch, bottomMargin=0.7 * inch, title='CS328 A2 Writeup').build(w)
print('built')
