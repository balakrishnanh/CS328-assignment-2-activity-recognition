"""Fill the FILL UP tables and interpretation cells of the executed notebook from results.json."""
import json, nbformat

R = json.load(open('results.json'))
nb = nbformat.read('A2_activity_recognition.ipynb', as_version=4)
p = lambda v: f'{100 * v:.1f}%'
W = {r['window_sec']: r for r in R['window_baseline']}
c1, c2, c3, c4 = R['c1'], R['c2'], R['c3'], R['c4']
rec = R['recordings']
ft = {r['set']: r for r in R['feature_table']}

def hp_dt(b): return f"max_depth={b['max_depth']}, criterion='{b['criterion']}'"
def hp_et(b): return (f"ExtraTrees(n_estimators={b['n_estimators']}, max_features={b['max_features']!r}, "
                      f"min_samples_leaf={b['min_samples_leaf']}, max_depth={b['max_depth']})")
def answer(idx, text):
    src = nb.cells[idx].source
    assert '**Write Below:**' in src, idx
    nb.cells[idx].source = src.split('**Write Below:**')[0] + '**Write Below:**\n\n' + text.strip() + '\n'

def table(idx, old_table, new_table):
    src = nb.cells[idx].source
    assert old_table in src, idx
    nb.cells[idx].source = src.replace(old_table, new_table)

# ---- Feature exploration (cell 19) ----
fin, base, old8, c8 = ft['FINAL (20)'], ft['baseline (7)'], ft['first-round 8 (magnitude only)'], ft['compact 8 = FEATURES[:8]']
s = nb.cells[19].source
start = s.index('| Window length'); end = s.index('#### 🛑')
table(19, s[start:end], f"""| Window length | Features | Accuracy (Class_Data test split) | Grouped CV, recordings held out (extra check) |
|-|-|-|-|
| 2 s | baseline | {p(W[2]['split_acc'])} | {p(W[2]['grouped_cv_acc'])} |
| 3 s | baseline | {p(W[3]['split_acc'])} | {p(W[3]['grouped_cv_acc'])} |
| 5 s | baseline | {p(W[5]['split_acc'])} | {p(W[5]['grouped_cv_acc'])} |
| 10 s | baseline | {p(W[10]['split_acc'])} | {p(W[10]['grouped_cv_acc'])} |
| 5 s | first round: 8 magnitude-only features | {p(old8['dt_split'])} | {p(old8['dt_grouped_cv'])} (tree), {p(old8['rf_grouped_cv'])} (random forest) |
| 5 s | compact: `FEATURES[:8]` | {p(c8['dt_split'])} | {p(c8['dt_grouped_cv'])} (tree), {p(c8['rf_grouped_cv'])} (random forest) |
| **5 s** | **final: `FEATURES` (20)** | **{p(R['window_final']['split_acc'])}** | {p(fin['dt_grouped_cv'])} (tree), **{p(fin['rf_grouped_cv'])}** (random forest) |

""")
answer(19, f"""
**Window length.** With the baseline features the depth-5 tree scores {p(W[2]['split_acc'])} / {p(W[3]['split_acc'])} / {p(W[5]['split_acc'])} / {p(W[10]['split_acc'])} at 2 / 3 / 5 / 10 s. That 3-point spread is about the noise of one random split, and the grouped CV (whole recordings held out) is flat at 48-50%. Window length is not what limits the model; the features are. We locked in **5 s**: about 9 steps per window (enough for autocorrelation and a 0.2 Hz FFT resolution), 612 of 615 recordings keep at least one window (10 s keeps 603), and a typical 40 s recording gets 5-10 votes.

**Features, in two rounds.** The baseline is 7 amplitude statistics of one signal, so the baseline tree never predicts the stairs (F1 0.00) and barely predicts lying (0.09).
*Round 1* added magnitude-only rhythm and noise features (step period, spectral entropy, peaks per second, power above 5 Hz). The random forest went from {p(base['rf_grouped_cv'])} to {p(old8['rf_grouped_cv'])} with recordings held out.
*Round 2* went back to the three axes, and that is where the large gain was (→ **{p(fin['rf_grouped_cv'])}**). Two new feature families did the work (F1 numbers below are for the depth-5 tree on the split):
- **Vertical proxy** (`vert_*`): project the 3-D motion onto its main axis, which is close to vertical while walking however the phone is rotated, then measure its period, spread, skew, kurtosis and stride regularity. `vert_period` is the single best addition (+{100 * (ft['baseline + vert_period']['rf_grouped_cv'] - base['rf_grouped_cv']):.1f} points grouped CV; upstairs F1 0→{ft['baseline + vert_period']['f1_upstairs']:.2f}, walk {ft['baseline + vert_period']['f1_walk']:.2f}).
- **What is left of orientation** (`mean_y/z`, `var_frac_y/z`, `slow_*`, `slow_fast_ratio`): Sensor Logger's gravity removal is imperfect, so the residual mean and the slow (<1 Hz) drift fall on the axes that are not vertical. These are the features that move lying (`mean_y`, `mean_z`, `var_frac_z`: F1 ≈ 0.2 each vs 0.09) and standing/sitting (`slow_fast_ratio`: standing → {ft['baseline + slow_fast_ratio']['f1_standing']:.2f}; `slow_z`: sitting → {ft['baseline + slow_z']['f1_sitting']:.2f}).

We chose the 20 by greedy forward selection with recordings held out. The first 8 alone (compact set) get {p(c8['rf_grouped_cv'])}, 12 get {p(ft['compact 12 = FEATURES[:12]']['rf_grouped_cv'])}, and all 20 get {p(fin['rf_grouped_cv'])}. We kept 20, above the suggested 5-8, because the remaining errors are in the hardest classes and every point there counts. Leave-one-out shows no single feature is critical (largest drop: `mean_y`, −{100 * (fin['rf_grouped_cv'] - ft['final - mean_y']['rf_grouped_cv']):.1f}): the information is spread over correlated features. Dropped: `rms` (duplicates `std`/`min`), `std_x/y/z` (replaced by variance fractions), `pca1`, `zcr`, `dom_freq` (harmonics make it jump between step and stride rate), and the round-1 features once the vertical-proxy versions were in.

**Still confused.** Lying vs sitting and upstairs vs walk/downstairs. Many static recordings are a phone lying still on a surface (see the failure analysis), so the signal has no posture in it at all.
""")

# ---- Classification 1 (cell 22) ----
table(22, """| Default (max_depth=5, criterion='entropy') | **FILL UP** | **FILL UP** |
| **FILL UP best hyperparameters**  | **FILL UP** | **FILL UP** |""",
f"""| Default (max_depth=5, criterion='entropy') | {p(c1['default']['split'])} | {p(c1['default']['test'])} |
| {hp_dt(c1['best_params'])} | {p(c1['best']['split'])} | {p(c1['best']['test'])} |""")
f1d = c3['f1_dt']
answer(22, f"""
**What the scores say.** The tree does well on the held-out 30% of Sample_Data ({p(c1['default']['split'])} default, {p(c1['best']['split'])} tuned) but only {p(c1['default']['test'])} / {p(c1['best']['test'])} on the class corpus. The split score is optimistic: Sample_Data is 35 recordings of 3-5 min from 5 people, so windows of the same recording and person sit on both sides of the split. Class_Data is a new device (phone vs wrist), new placements and about 100 new people. Wrist and pocket differ physically: the wrist device's noise floor at rest is about 3x higher, and wrist walking is gentler. And the axis features mean something different on a wrist than in a pocket, so a tree that splits on `mean_z` or `var_frac_y` learns wrist-specific rules. With the first-round magnitude-only features the same tree transferred at 45.6%. The orientation features help on phones and hurt in this one direction.

**Which activities are distinguishable.** Jogging transfers (F1 {f1d['jogging']:.2f}): it is 3-4x more intense than anything else on any device. Everything else is between 0.22 and 0.43 F1: walk {f1d['walk']:.2f}, upstairs {f1d['upstairs']:.2f}, downstairs {f1d['downstairs']:.2f}, standing {f1d['standing']:.2f}, sitting {f1d['sitting']:.2f}, lying {f1d['lying']:.2f}.

**Hyperparameter tuning.** The grid prefers a deeper tree ({hp_dt(c1['best_params'])}: split {p(c1['default']['split'])} → {p(c1['best']['split'])}), but the class-corpus accuracy *drops* {100 * (c1['default']['test'] - c1['best']['test']):.1f} points. Tuning on a leaky split of five people fits those five people better and generalises worse. Five subjects are too few to learn a 7-class model that transfers.
""")

# ---- Classification 2 (cell 25) ----
table(25, """| Default (max_depth=5, criterion='entropy') | **FILL UP** | **FILL UP** |
| **FILL UP best hyperparameters**  | **FILL UP** | **FILL UP** |""",
f"""| Default (max_depth=5, criterion='entropy') | {p(c2['default']['split'])} | {p(c2['default']['test'])} |
| {hp_dt(c2['best_params'])} | {p(c2['best']['split'])} | {p(c2['best']['test'])} |""")
f1d = c4['f1_dt']
answer(25, f"""
**What the scores say.** Trained on the class corpus, the tree reaches {p(c2['default']['split'])} (default) / {p(c2['best']['split'])} (tuned) on its own split and {p(c2['default']['test'])} / {p(c2['best']['test'])} on Sample_Data. Transfer is much better than in the other direction: 612 recordings from many phones, placements and people cover far more variation than 35 recordings from 5 wrists. The in-domain split is low for a single tree because Class_Data is noisy: recordings range from 4 s to 27 min, phones sit in pockets, hands or on tables, and some labels are loose (a "walk" file named "cooking").

**Which activities are distinguishable.** On Sample_Data, jogging is nearly perfect (F1 {f1d['jogging']:.2f}). Standing ({f1d['standing']:.2f}), sitting ({f1d['sitting']:.2f}), walk ({f1d['walk']:.2f}) and downstairs ({f1d['downstairs']:.2f}) are partly right. Upstairs ({f1d['upstairs']:.2f}) and lying ({f1d['lying']:.2f}) are worst.

**Hyperparameter tuning.** The split picks {hp_dt(c2['best_params'])} (+{100 * (c2['best']['split'] - c2['default']['split']):.1f} points on the split), yet Sample_Data accuracy *drops* from {p(c2['default']['test'])} to {p(c2['best']['test'])}. The random split leaks windows of the same recording into the test set, so it rewards deep trees that memorise recordings. A grouped CV agrees with the transfer result: depth 5 gets {p(R['models'][0]['grouped_cv'])} vs {p(R['models'][1]['grouped_cv'])} for depth {c2['best_params']['max_depth']} (Model Exploration cell). Lesson: tune on a split that holds out whole recordings, or people.
""")

# ---- Classification 3 (cell 28) ----
table(28, """| **FILL UP best hyperparameters**  | **FILL UP** | **FILL UP** |""",
      f"""| {hp_et(c3['best_params'])} | {p(c3['best']['split'])} | {p(c3['best']['test'])} |""")
g3 = {c: c3['f1_rf'][c] - c3['f1_dt'][c] for c in c3['f1_dt']}
answer(28, f"""
**Extra Trees vs decision tree (Sample_Data → Class_Data).** Extra Trees is a random forest that also picks its split thresholds at random. It beats the best tree on the Sample_Data split ({p(c3['best']['split'])} vs {p(c1['best']['split'])}) and on the class corpus ({p(c3['best']['test'])} vs {p(c1['best']['test'])}, +{100 * (c3['best']['test'] - c1['best']['test']):.1f} points). Averaging {c3['best_params']['n_estimators']} randomised trees smooths out the thresholds a single tree overfits to five people, which matters most in this direction.

**Where the gain is.** Mostly walk ({g3['walk']:+.2f} F1) and sitting ({g3['sitting']:+.2f}), then jogging ({g3['jogging']:+.2f}) and standing ({g3['standing']:+.2f}). Upstairs ({g3['upstairs']:+.2f}), downstairs ({g3['downstairs']:+.2f}) and lying ({g3['lying']:+.2f}) do not improve: trained on 5 wrists, no model learns the subtle cues for those classes. The ~48% here is the ceiling of this direction, not a modelling problem.
""")

# ---- Classification 4 (cell 31) ----
table(31, """| **FILL UP best hyperparameters**  | **FILL UP** | **FILL UP** |""",
      f"""| {hp_et(c4['best_params'])} | {p(c4['best']['split'])} | {p(c4['best']['test'])} |""")
imp = sorted(c4['importance'].items(), key=lambda kv: -kv[1])
answer(31, f"""
**Extra Trees vs decision tree (Class_Data → Sample_Data).** Extra Trees wins clearly: {p(c4['best']['split'])} vs {p(c2['best']['split'])} on the class split and **{p(c4['best']['test'])}** vs {p(c2['best']['test'])} on Sample_Data (+{100 * (c4['best']['test'] - c2['best']['test']):.1f} points). Its most important features are `{imp[0][0]}` ({imp[0][1]:.2f}), `{imp[1][0]}` ({imp[1][1]:.2f}) and `{imp[2][0]}` ({imp[2][1]:.2f}): overall intensity first, then the slow-drift and vertical-proxy features.

**Did it help all classes equally? No.** The gain is spread over five classes: walk {c4['f1_dt']['walk']:.2f}→{c4['f1_rf']['walk']:.2f}, upstairs {c4['f1_dt']['upstairs']:.2f}→{c4['f1_rf']['upstairs']:.2f}, downstairs {c4['f1_dt']['downstairs']:.2f}→{c4['f1_rf']['downstairs']:.2f}, standing {c4['f1_dt']['standing']:.2f}→{c4['f1_rf']['standing']:.2f}, sitting {c4['f1_dt']['sitting']:.2f}→{c4['f1_rf']['sitting']:.2f}. Jogging was already solved ({c4['f1_rf']['jogging']:.2f}). Lying gets **worse** ({c4['f1_dt']['lying']:.2f}→{c4['f1_rf']['lying']:.2f}). The ensemble votes a still wrist into "sitting" or "standing", most likely because in the class corpus a still phone is more often a sitting recording than a lying one. That gap is in the data (no posture signal), not in the model.
""")

# ---- Classification 5 (cell 38) ----
r = {(x['model'].split()[0], x['direction'].split()[0]): x for x in rec}
table(38, """| Decision Tree (best) | Class_Data → Sample_Data | **FILL UP** | **FILL UP** |
| **FILL UP your new classifier** (best) | Class_Data → Sample_Data | **FILL UP** | **FILL UP** |
| Decision Tree (best) | Sample_Data → Class_Data | **FILL UP** | **FILL UP** |
| **FILL UP your new classifier** (best) | Sample_Data → Class_Data | **FILL UP** | **FILL UP** |""",
f"""| Decision Tree (best) | Class_Data → Sample_Data | {p(r['Decision', 'Class_Data']['window_acc'])} | {p(r['Decision', 'Class_Data']['recording_acc'])} |
| Extra Trees (best) | Class_Data → Sample_Data | {p(r['Extra', 'Class_Data']['window_acc'])} | **{p(r['Extra', 'Class_Data']['recording_acc'])}** |
| Decision Tree (best) | Sample_Data → Class_Data | {p(r['Decision', 'Sample_Data']['window_acc'])} | {p(r['Decision', 'Sample_Data']['recording_acc'])} |
| Extra Trees (best) | Sample_Data → Class_Data | {p(r['Extra', 'Sample_Data']['window_acc'])} | {p(r['Extra', 'Sample_Data']['recording_acc'])} |""")
et = r['Extra', 'Class_Data']; dtc = r['Decision', 'Class_Data']
answer(38, f"""
**Does aggregating help?** Yes, for every model and direction: +{100 * (dtc['recording_acc'] - dtc['window_acc']):.1f} points for the tree and +{100 * (et['recording_acc'] - et['window_acc']):.1f} for Extra Trees on Sample_Data, and +1-2 points on Class_Data. A majority vote amplifies whatever the window classifier does on a recording. When most windows are right, stray errors are outvoted: for Extra Trees on Sample_Data, sitting goes {et['per_class_window']['sitting']:.0%} → {et['per_class_recording']['sitting']:.0%} of recordings, standing {et['per_class_window']['standing']:.0%} → {et['per_class_recording']['standing']:.0%}, downstairs {et['per_class_window']['downstairs']:.0%} → {et['per_class_recording']['downstairs']:.0%}.

**When majority vote makes things worse.** (1) A class whose windows are right less than half the time: the vote turns a partial score into zero (lying: {et['per_class_window']['lying']:.0%} of windows, 0% of recordings). (2) Short recordings: a 10 s recording has 1-2 windows, so one bad window decides it. (3) Window accuracy is dominated by a few very long recordings (one 27-minute sitting file alone gives 323 windows), while recording accuracy counts each recording once. (4) Recordings that mix activities, such as handling the phone at the start, can be outvoted toward the wrong label. We also tried confidence weighting (summing `predict_proba` over a recording): {p(et['confidence_vote_acc'])} vs {p(et['recording_acc'])} for Extra Trees and {p(dtc['confidence_vote_acc'])} vs {p(dtc['recording_acc'])} for the tree on Sample_Data. Majority vote won, so we kept it and use summed probability only to break ties.
""")

nbformat.write(nb, 'A2_activity_recognition.ipynb')
print('filled')
