"""Build exploration_report.pdf and writeup.pdf from results.json and figures/ (run after the notebook)."""
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

R = json.load(open('results.json'))
N = sum(1 for r in R['feature_table'] if r['set'].startswith('baseline + ') and r['set'] != 'baseline + all candidates')
CLASSES = R['final']['classes']
ft = {r['set']: r for r in R['feature_table']}
base, fin = ft['baseline (7)'], ft['FINAL (20)']
old8, c8, c12 = ft['first-round 8 (magnitude only)'], ft['compact 8 = FEATURES[:8]'], ft['compact 12 = FEATURES[:12]']
fp, c4 = R['final'], R['c4']
cm = fp['cm']                                   # rows = true, cols = predicted
M = {m['model'].split(' (')[0].split(' {')[0]: m for m in R['models']}
p = lambda v: f'{100 * v:.1f}%'
pts = lambda v: f'{100 * v:+.1f}'
rec = {(x['model'].split()[0], x['direction'].split()[0]): x for x in R['recordings']}

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
    st = [('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#999999')), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
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

s = []
s.append(P('CS328 A2 — Activity Classification: Exploration Report', h1))
s.append(P('Data: 615 Class_Data recordings (Sensor Logger phones, gravity removed) and 35 Sample_Data recordings '
           '(wrist device, 5 people). Every number comes from the submitted notebook (<i>results.json</i>). '
           '"Split" is the notebook\'s 70/30 random window split. Because it puts windows of the same recording on both sides, '
           'we also report <b>grouped CV</b>: 5-fold CV on Class_Data with whole recordings held out, which is what a new user '
           'would see. We chose features and models on grouped CV, not on the leaky split.', body))

# ------------------------------------------------------------------ §1
s.append(P('§1 Feature exploration', h2))
wb = R['window_baseline']
s.append(P(f'<b>Window length</b> (baseline features, depth-5 entropy tree): 2 s {p(wb[0]["split_acc"])}, 3 s {p(wb[1]["split_acc"])}, '
           f'5 s {p(wb[2]["split_acc"])}, 10 s {p(wb[3]["split_acc"])} on the split, and 48-50% grouped CV at every length, so window '
           'length is not the bottleneck. We chose <b>5 s, non-overlapping</b>: about 9 steps per window, 612 of 615 recordings keep at '
           'least one window (10 s keeps 603), and a typical 40 s recording gets 5-10 votes.', body))
s.append(P(f'<b>Two rounds of features.</b> Round 1 used only the magnitude (rhythm, spectrum, noise): grouped CV '
           f'{p(base["rf_grouped_cv"])} → {p(old8["rf_grouped_cv"])}. Round 2 went back to the three axes and added a <i>vertical proxy</i> '
           f'(the motion projected on its main axis, close to vertical while walking whatever the phone\'s rotation) and features of what '
           f'Sensor Logger\'s imperfect gravity removal leaves behind (residual mean and slow &lt;1 Hz drift per axis): → '
           f'<b>{p(fin["rf_grouped_cv"])}</b>. Each row below adds one feature to the 7 baseline statistics. "Change, split" = depth-5 tree on the '
           f'notebook split; "Change, grouped CV" = random forest (100 trees) with recordings held out; "Drop from final" = grouped-CV change '
           f'when the feature is removed from the final 20 ({p(fin["rf_grouped_cv"])}). F1 values are the depth-5 tree\'s.', body))

why = {
    'min': ('static vs gait', '(baseline) Deepest dip of the detrended magnitude: large on every step, ≈0 at rest. Top importance (0.11).'),
    'vert_period': ('upstairs (F1 0→0.28), walk', 'Lag of the strongest autocorrelation peak of the vertical proxy: a full stride, slower on stairs.'),
    'vert_std': ('jogging (0.79→0.87), downstairs', 'Spread of vertical motion: bounce height. Rotation invariant, unlike std_z.'),
    'slow_fast_ratio': ('standing (0.48→0.52), downstairs', 'Slow (&lt;1 Hz) drift vs fast (&gt;5 Hz) jitter. Standing sways; a still phone only has sensor noise.'),
    'vert_stride_reg': ('lying (0.09→0.21), downstairs', 'Autocorrelation at the second peak: how alike consecutive strides are (Moe-Nilssen regularity).'),
    'slow_z': ('sitting (0.65→0.70), standing', 'Slow drift on the z axis: gravity-removal residual, so it depends on how the phone is tilted.'),
    'var_frac_y': ('upstairs (0→0.19)', 'Share of the motion on the y axis: where the motion goes relative to the phone.'),
    'kurt': ('none alone', 'Spiky (impacts, taps) vs smooth magnitude.'),
    'mean_z': ('lying (0.09→0.19), downstairs', 'Residual gravity left on z after Sensor Logger subtracts its gravity estimate: a weak tilt cue.'),
    'acf_peak': ('downstairs (0→0.53)', 'Regularity of the magnitude: periodic gait vs irregular motion.'),
    'mean_y': ('lying (0.09→0.21)', 'Residual gravity on y. Largest single drop when removed from the final set.'),
    'var_frac_z': ('lying (0.09→0.21)', 'Share of the motion on z.'),
    'vert_kurt': ('upstairs (0→0.34)', 'Peakedness of vertical acceleration: sharp landings vs smooth push-off.'),
    'logvar': ('none alone', 'Log of the magnitude variance: intensity on a log scale (rest spans decades). Second-highest importance.'),
    'pca2': ('lying (0.09→0.19)', 'Share of motion on the 2nd principal axis: flat vs line-like motion.'),
    'vert_skew': ('downstairs (0→0.42)', 'Asymmetry of vertical acceleration: push-off up vs landing down.'),
    'slow_frac_y': ('none alone', 'Share of slow drift on y; helps only together with slow_z.'),
    'slow_frac_z': ('none alone', 'Share of slow drift on z.'),
    'med': ('none alone', '(baseline) Median of the detrended magnitude.'),
    'corr_yz': ('downstairs (0→0.42)', 'Correlation of the y and z axes: coordinated vs independent motion.'),
}
dropped = {
    'step_period': 'Round-1 magnitude version of vert_period; superseded.',
    'spec_entropy': 'Round 1, +3.5 alone, but redundant with acf_peak + vert_* once those are in.',
    'hf_frac': 'Round 1; covered by slow_fast_ratio.',
    'std_z': 'Depends on placement; replaced by variance fractions.',
    'dom_freq': 'Harmonics make it jump between step and stride rate.',
    'rms': 'Duplicates std and min.',
}
rows = [['Feature', 'Class it helps discriminate', 'Change, split', 'Change, grouped CV', 'Drop from final', 'Rationale']]
for f, (cls, txt) in why.items():
    r = base if f in ('min', 'med') else ft[f'baseline + {f}']
    d = ft[f'final - {f}']
    rows.append([f'<b>{f}</b>', cls, '—' if f in ('min', 'med') else pts(r['dt_split'] - base['dt_split']),
                 '—' if f in ('min', 'med') else pts(r['rf_grouped_cv'] - base['rf_grouped_cv']),
                 pts(d['rf_grouped_cv'] - fin['rf_grouped_cv']), txt])
for f, txt in dropped.items():
    r = ft[f'baseline + {f}']
    rows.append([f + ' (dropped)', '', pts(r['dt_split'] - base['dt_split']), pts(r['rf_grouped_cv'] - base['rf_grouped_cv']), '—', txt])
for name, r in [('round-1 8 (magnitude only)', old8), ('compact 8 = first 8 above', c8), ('compact 12', c12), ('<b>final 20</b>', fin),
                (f'baseline + all {N}', ft['baseline + all candidates'])]:
    rows.append([name, '', pts(r['dt_split'] - base['dt_split']), pts(r['rf_grouped_cv'] - base['rf_grouped_cv']), '',
                 f'grouped CV {p(r["rf_grouped_cv"])}'])
s.append(T(rows, [0.95 * inch, 1.3 * inch, 0.5 * inch, 0.6 * inch, 0.5 * inch, 3.4 * inch], font=7))
s.append(Spacer(1, 3))
s.append(P(f'<b>How we chose.</b> Greedy forward selection over 90 research candidates with recordings held out (research/exp15.py), then '
           f're-implemented per window in add_my_features ({N} candidates kept there); the order '
           'above is the order of selection. The rubric suggests 5-8 features: the first 8 alone reach '
           f'{p(c8["rf_grouped_cv"])}, and we kept 20 ({p(fin["rf_grouped_cv"])}) because the remaining errors are in the hardest '
           'classes and every point there counts. <b>Honest note:</b> leave-one-out changes grouped CV by ≤1.6 points for every '
           'feature (several within ±0.5, i.e. fold noise), so the information is spread across correlated features rather than '
           f'carried by any one of them. Adding all {N} candidates is <i>worse</i> than our 20 (noise features dilute the forest).', body))

# ------------------------------------------------------------------ §2
s.append(P('§2 Model exploration', h2))
s.append(P('Same 20 features and 5 s windows for every model. Tree and Extra Trees hyperparameters come from the notebook grid searches '
           '(chosen on the Class_Data split); the others are standard settings.', body))
bp4 = c4['best_params']
rows = [['Model', 'Key hyperparameter(s)', 'Split', 'Grouped CV', 'Class→Sample', 'Sample→Class', 'F1 up / lying', 'Notes']]
spec = [('Decision Tree', 'max_depth=5, entropy', 'Baseline, interpretable; almost never predicts upstairs.'),
        ('Decision Tree', f'max_depth={R["c2"]["best_params"]["max_depth"]}, {R["c2"]["best_params"]["criterion"]}', 'Grid winner on the split, worse on grouped CV and transfer: memorises recordings.'),
        ('Random Forest', 'n_estimators=300', 'Close second; best Class→Sample and Sample→Class.'),
        ('Extra Trees', f'n_estimators={bp4["n_estimators"]}, max_features={bp4["max_features"]}, max_depth={bp4["max_depth"]}', '<b>Chosen.</b> Best split and grouped CV; random thresholds average out person-to-person noise.'),
        ('HistGradientBoosting', 'lr=0.05, 300 iterations', 'Best upstairs/lying F1, but overfits the leaky split more than it gains.'),
        ('Gradient Boosting', 'n_estimators=100, max_depth=3', 'Slower, weaker version of the above.'),
        ('kNN', 'k=15, StandardScaler', 'Distances across 20 mixed-scale features are noisy; worst transfer.'),
        ('Logistic Regression', 'C=1, StandardScaler', 'Linear boundaries: fine for intensity, not for static classes.'),
        ('SVM', 'RBF, C=3, StandardScaler', 'Good in-domain, worst Sample→Class.')]
for (name, hp, note), m in zip(spec, R['models']):
    rows.append([name, hp, p(m['split']), p(m['grouped_cv']), p(m['C->S']), p(m['S->C']),
                 f"{m['f1_upstairs']:.2f} / {m['f1_lying']:.2f}", note])
s.append(T(rows, [0.95 * inch, 1.35 * inch, 0.45 * inch, 0.55 * inch, 0.6 * inch, 0.6 * inch, 0.55 * inch, 2.2 * inch]))
s.append(Spacer(1, 3))
et_m = R['models'][3]; dt_m = R['models'][0]
s.append(P(f'<b>Which won and why.</b> The tree ensembles (Extra Trees {p(et_m["grouped_cv"])}, random forest {p(R["models"][2]["grouped_cv"])}, '
           f'boosting 62-63%) and the RBF SVM (62%) form a top group about 13 points above the single tree ({p(dt_m["grouped_cv"])}). '
           f'We picked <b>Extra Trees</b>: best on the notebook split ({p(et_m["split"])}) and on grouped CV, and within a point of the '
           'best on Class→Sample. With 20 features a single tree has to spend each split on one threshold, so classes that need '
           'several weak cues together (upstairs: slower vertical period plus kurtosis plus variance share; lying: residual gravity '
           'on two axes) get few leaves; hundreds of randomised trees cover those regions. <b>The gain was not equal across '
           f'classes.</b> Class→Sample, Extra Trees raised upstairs F1 {c4["f1_dt"]["upstairs"]:.2f}→{c4["f1_rf"]["upstairs"]:.2f}, '
           f'downstairs {c4["f1_dt"]["downstairs"]:.2f}→{c4["f1_rf"]["downstairs"]:.2f}, walk {c4["f1_dt"]["walk"]:.2f}→{c4["f1_rf"]["walk"]:.2f}, '
           f'sitting {c4["f1_dt"]["sitting"]:.2f}→{c4["f1_rf"]["sitting"]:.2f}, standing {c4["f1_dt"]["standing"]:.2f}→{c4["f1_rf"]["standing"]:.2f}; '
           f'jogging stayed at {c4["f1_rf"]["jogging"]:.2f}, and lying got worse ({c4["f1_dt"]["lying"]:.2f}→{c4["f1_rf"]["lying"]:.2f}): the ensemble '
           'votes a still wrist into "sitting", the more common still class in the training data. The tuned-tree row shows the cost of '
           'tuning on the leaky split: the deeper tree wins the split (+6.9) but loses on grouped CV (−2.2) and on Sample_Data (−6.1).', body))

# ------------------------------------------------------------------ §3
s.append(P('§3 Universal pipeline', h2))
s.append(T([
    ['Step', 'Choice'],
    ['Preprocessing', 'Sampling rate per file = 1 / median timestamp step (50-101 Hz in the data). Magnitude of (x,y,z) detrended by the '
                      'recording mean (calc_magnitude); 2nd-order Butterworth low-pass at 5 Hz, zero-phase (remove_noise).'],
    ['Windows', '5 s, non-overlapping, clock-aligned via DataFrame.resample; windows with &lt;80% of the expected samples are dropped.'],
    ['Features (20)', 'min, vert_period, vert_std, slow_fast_ratio, vert_stride_reg, slow_z, var_frac_y, kurt, mean_z, acf_peak, mean_y, '
                      'var_frac_z, vert_kurt, logvar, pca2, vert_skew, slow_frac_y, slow_frac_z, med, corr_yz (definitions in add_my_features)'],
    ['Classifier', f'ExtraTreesClassifier(n_estimators={bp4["n_estimators"]}, max_features={bp4["max_features"]}, '
                   f'min_samples_leaf={bp4["min_samples_leaf"]}, max_depth={bp4["max_depth"]}, random_state=42), trained on Class_Data'],
    ['Aggregation', 'Majority vote of the window labels per recording (recording = activity folder + file name, because Sample_Data reuses '
                    '026.csv…030.csv in every folder); ties broken by the summed predict_proba.'],
    ['Performance', f'Notebook split <b>{p(c4["best"]["split"])}</b>. Grouped CV on Class_Data: <b>{p(fp["grouped_cv_window_acc"])}</b> of windows, '
                    f'<b>{p(fp["grouped_cv_recording_acc"])}</b> of recordings. Class→Sample: {p(c4["best"]["test"])} of windows, '
                    f'<b>{p(rec["Extra", "Class_Data"]["recording_acc"])}</b> of recordings.'],
], [1.1 * inch, 6.1 * inch]))
s.append(Spacer(1, 3))
s.append(P('<b>How we decided.</b> One axis at a time. Window length first, with baseline features, because the grouped CV showed it '
           'barely mattered. Features next, judged with a random forest under grouped CV: the 30% random split leaks windows of the '
           'same recording into test and kept preferring deep trees that then did worse on new recordings and on Sample_Data. '
           'Round 1 (magnitude only) plateaued at 57%; the error analysis said the confusions were posture and stairs, which live in '
           'axis information, so round 2 added the vertical proxy and gravity-residual features. Model last: Extra Trees was best or '
           'tied-best on every protocol except Sample→Class, where nothing reaches 49%. For aggregation, majority vote beat confidence '
           f'weighting ({p(rec["Extra", "Class_Data"]["recording_acc"])} vs {p(rec["Extra", "Class_Data"]["confidence_vote_acc"])} Class→Sample).', body))

# ------------------------------------------------------------------ §4
s.append(P('§4 Confusion matrix + per-class analysis', h2))
s.append(P(f'Final pipeline, Class_Data, grouped 5-fold CV, window level ({sum(map(sum, cm))} windows, accuracy '
           f'{p(fp["grouped_cv_window_acc"])}). Rows = predicted, columns = actual, as in the template.', body))
rows = [['Predicted ↓ / Actual →'] + CLASSES]
for j, c in enumerate(CLASSES):
    rows.append([f'<b>{c}</b>'] + [f'<b>{cm[i][j]}</b>' if i == j else str(cm[i][j]) for i in range(7)])
rows.append(['recall'] + [f'{cm[i][i] / sum(cm[i]):.0%}' for i in range(7)])
s.append(T(rows, [1.3 * inch] + [0.8 * inch] * 7))
rc = fp['rec_cm']
s.append(P(f'Recording level (majority vote, same CV): {p(fp["grouped_cv_recording_acc"])}; per-class recall '
           + ', '.join(f'{c} {rc[i][i] / max(1, sum(rc[i])):.0%}' for i, c in enumerate(CLASSES)) + '.', small))
s.append(Spacer(1, 3))
md = fp['medians']
an = [
    ['Cell (actual → predicted)', 'Count', 'Cause', 'Axis'],
    ['lying → sitting', cm[6][5], f'A still phone reads ≈0 whatever the posture. 43 of 82 lying and 68 of 116 sitting recordings have &lt;0.01 m/s² of slow motion, i.e. the phone is resting (research/exp14.py). Medians: vert_std {md["vert_std"]["lying"]:.3f} vs {md["vert_std"]["sitting"]:.3f}, logvar {md["logvar"]["lying"]:.2f} vs {md["logvar"]["sitting"]:.2f}.', 'Feature'],
    ['standing → sitting', cm[4][5], f'Standing usually sways (median logvar {md["logvar"]["standing"]:.1f} vs {md["logvar"]["sitting"]:.1f}), but a phone put down on a table while the person stands is indistinguishable from sitting.', 'Feature'],
    ['upstairs → walk', cm[1][0], f'Same intensity and cadence: min {md["min"]["upstairs"]:.2f} vs {md["min"]["walk"]:.2f}, vert_std {md["vert_std"]["upstairs"]:.2f} vs {md["vert_std"]["walk"]:.2f}, vert_period {md["vert_period"]["upstairs"]:.2f} vs {md["vert_period"]["walk"]:.2f} s.', 'Feature'],
    ['sitting → lying', cm[5][6], 'The mirror of lying→sitting; the forest splits the "phone at rest" recordings between the two classes on weak residual-gravity cues.', 'Feature'],
    ['sitting → standing', cm[5][4], 'Fidgeting, or holding the phone while sitting, adds sway-like slow motion.', 'Feature'],
    ['downstairs → walk / upstairs', f'{cm[2][0]} / {cm[2][1]}', 'A slow, careful descent loses the sharp landings (vert_kurt, vert_skew) that set downstairs apart; the magnitude has no sign of up vs down.', 'Feature'],
    ['upstairs → downstairs', cm[1][2], 'Both stairs are slower and more irregular than walking; telling up from down needs the direction of vertical motion, which is lost with gravity.', 'Feature'],
    ['lying → standing', cm[6][4], 'Recordings with the phone held in the hand while lying: hand tremor looks like standing sway.', 'Feature'],
    ['walk → down / up', f'{cm[0][2]} / {cm[0][1]}', 'Heavy or brisk walkers produce stair-like impacts. The single tree confuses far more of these than the forest (walk F1 0.60 vs 0.76), so for the tree this part is model side.', 'Feature (model for DT)'],
]
s.append(T(an, [1.35 * inch, 0.5 * inch, 4.55 * inch, 0.8 * inch]))
s.append(P('Overall: the remaining error is on the <b>feature side</b>. The five best model families top out at 62-66% grouped CV, and the confused '
           'pairs have nearly equal medians. The clear model-side effect is the single tree vs the ensembles (§2): about 13 points, '
           'mostly on walk and the stairs.', body))

# ------------------------------------------------------------------ §5
s.append(P('§5 Top failure modes + edge cases', h2))
f1, f2 = fp['failures']
def k(f, w): return f'median {f[w]} ({"true" if w == "true" else "predicted"})'
v1, v2 = f1['values'], f2['values']
s.append(P(f'<b>Failure 1: lying classified as sitting</b> ({cm[6][5]} windows, {cm[6][5] / sum(cm[6]):.0%} of all lying windows; only '
           f'{cm[6][6] / sum(cm[6]):.0%} of lying is right). Example: <i>lying/{f1["recording"]}</i>, {f1["hits"]}/{f1["n"]} windows → sitting.', body))
s.append(img('figures/failure_1.png', 7.2 * inch))
s.append(P(f'Left: the failing trace (blue) is flat: the phone is resting. A correctly classified lying recording (orange) for comparison; '
           f'red bands = windows predicted wrong. Right: the failing window\'s features vs the class medians, each row scaled by its max. '
           f'vert_std {v1["failing window"]["vert_std"]:.3f} (lying {v1[k(f1, "true")]["vert_std"]:.3f}, sitting {v1[k(f1, "pred")]["vert_std"]:.3f}), '
           f'slow_z {v1["failing window"]["slow_z"]:.3f} ({v1[k(f1, "true")]["slow_z"]:.3f} / {v1[k(f1, "pred")]["slow_z"]:.3f}): the failing window is '
           'just as close to sitting as to lying.', cap))
s.append(P('<b>Mechanism:</b> Sensor Logger subtracts gravity, so the one thing that differs between lying and sitting, which way the '
           'body (and phone) points, is not in the signal. When the phone rests on a bed or table, all that is left is sensor noise. '
           '<b>Feature problem:</b> the two classes have the same feature distribution. <b>Fix:</b> also log Sensor Logger\'s gravity '
           'sensor (or the raw accelerometer) and add the tilt of the gravity vector in phone coordinates.', body))
s.append(P(f'<b>Failure 2: standing classified as sitting</b> ({cm[4][5]} windows, {cm[4][5] / sum(cm[4]):.0%} of standing windows). '
           f'Example: <i>standing/{f2["recording"]}</i>, {f2["hits"]}/{f2["n"]} windows → sitting.', body))
s.append(img('figures/failure_2.png', 7.2 * inch))
s.append(P(f'The failing recording (blue) is almost perfectly flat: logvar {v2["failing window"]["logvar"]:.1f} vs a standing median of '
           f'{v2[k(f2, "true")]["logvar"]:.1f} and a sitting median of {v2[k(f2, "pred")]["logvar"]:.1f}; vert_std {v2["failing window"]["vert_std"]:.3f} '
           f'vs {v2[k(f2, "true")]["vert_std"]:.3f} / {v2[k(f2, "pred")]["vert_std"]:.3f}. A person standing with the phone in a pocket sways '
           '(orange); this phone was not on a body.', cap))
s.append(P('<b>Mechanism:</b> the phone was set down (or held perfectly still) while the participant stood, so the recording contains '
           'no body sway at all, which is the only cue for standing in gravity-free data. Of all still recordings, sitting is the most '
           'common label, so the model calls it sitting. <b>Feature problem</b> (arguably a labelling/protocol problem). <b>Fix:</b> require '
           'a fixed placement (front trouser pocket) in the study protocol, or add the gravity-vector tilt, which separates upright '
           'from seated thighs.', body))
s.append(P(f'<b>Third largest, for completeness: upstairs → walk</b> ({cm[1][0]} windows, {cm[1][0] / sum(cm[1]):.0%}). Climbing at a natural '
           'pace has the same cadence and magnitude as walking; the difference (lifting the body each step) is on the vertical axis, '
           'and the vertical proxy only recovers part of it. Fix: barometer (a flight ≈ 0.3 hPa) or projecting on the gravity vector.', body))
s.append(P('<b>Edge cases in the data:</b> recordings from 4.5 s to 27 min (the 27-min sitting file alone gives 323 windows); a few phones at '
           '50-58 Hz, hence the per-file sampling-rate estimate; one lying file with a 20 s logging gap (those windows are skipped); a '
           '"walk" recording named "activity_1_-_cooking"; 3 recordings shorter than 5 s produce no window.', body))
s.append(P(f'<b>Why not 80%?</b> We tried: 90 candidate features, 9 model families, 2-10 s windows, trimming recording ends, per-recording '
           'weighting and recording-level context features (research/exp6, exp9-18). The ceiling on unseen recordings is about '
           f'66-67%, and our numbers match published subject-independent results: UniMiB-SHAR reports 73% (9 classes) and 57% (17 classes) '
           'with leave-subject-out vs much higher with random splits. ExtraSensory reports leave-user-out per-class scores from 0.63 '
           '(sitting) to 0.77 (walking). On the leaky notebook split the same pipeline scores 78%; that gap is the leakage, not skill.', body))
s.append(P('References: Bao &amp; Intille 2004 (window features + decision trees for accelerometer HAR); Kwapisz, Weiss &amp; Moore 2011, '
           'WISDM (phone in pocket; stairs hardest); Anguita et al. 2013, UCI-HAR (time/frequency features, gravity-separated signals); '
           'Moe-Nilssen &amp; Helbostad 2004 (gait regularity from trunk-acceleration autocorrelation); Micucci et al. 2017, UniMiB-SHAR '
           '(arxiv.org/abs/1611.07688); Vaizman et al. 2018, ExtraSensory (arxiv.org/abs/1609.06354).', small))

SimpleDocTemplate('exploration_report.pdf', pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                  topMargin=0.55 * inch, bottomMargin=0.55 * inch, title='CS328 A2 Exploration Report').build(s)

# ------------------------------------------------------------------ writeup
w = [P('CS328 A2 — Writeup', h1)]
w.append(P('<b>Most important decision.</b> Going back to the three axes after the magnitude-only features stalled at 57%. A vertical '
           'proxy (motion projected on its main axis) and the residue Sensor Logger\'s gravity removal leaves on each axis raised '
           f'accuracy on unseen recordings to {p(fp["grouped_cv_window_acc"])} (recordings: {p(fp["grouped_cv_recording_acc"])}). Judging every '
           'choice with whole recordings held out, not the leaky window split, is what told us which changes were real.', body))
w.append(P(f'<b>Biggest surprise in the confusion matrix.</b> Lying is called sitting {cm[6][5] / sum(cm[6]):.0%} of the time and is right '
           f'only {cm[6][6] / sum(cm[6]):.0%} of the time. Looking at the traces, about half the lying and sitting recordings are a phone '
           'resting completely still, so they contain no posture information at all, and no model can fix that.', body))
w.append(P('<b>Next.</b> Log the gravity vector and barometer, add tilt and altitude-change features for lying/sitting and stairs, and '
           'evaluate leave-one-person-out once participant IDs are available.', body))
w.append(P('<b>Contribution statement.</b> Harish Balakrishnan: set up the group repo and uploaded the starter notebook, util.py '
           'and class data. Vedant Naidu: data loading and windowing, feature engineering, model comparison and tuning, '
           'failure analysis, exploration report and writeup.', body))
SimpleDocTemplate('writeup.pdf', pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                  topMargin=0.7 * inch, bottomMargin=0.7 * inch, title='CS328 A2 Writeup').build(w)
print('built')
