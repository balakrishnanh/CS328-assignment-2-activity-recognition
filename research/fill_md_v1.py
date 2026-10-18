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
def hp_rf(b): return f"RandomForest(n_estimators={b['n_estimators']}, max_depth={b['max_depth']}, min_samples_leaf={b['min_samples_leaf']})"
def answer(idx, text):
    src = nb.cells[idx].source
    assert '**Write Below:**' in src, idx
    nb.cells[idx].source = src.split('**Write Below:**')[0] + '**Write Below:**\n\n' + text.strip() + '\n'

def table(idx, old_table, new_table):
    src = nb.cells[idx].source
    assert old_table in src, idx
    nb.cells[idx].source = src.replace(old_table, new_table)

# ---- Feature exploration (cell 19) ----
s = nb.cells[19].source
start = s.index('| Window length'); end = s.index('#### 🛑')
table(19, s[start:end], f"""| Window length | Features | Accuracy (Class_Data test split) | Grouped CV, recordings held out (extra check) |
|-|-|-|-|
| 2 s | baseline | {p(W[2]['split_acc'])} | {p(W[2]['grouped_cv_acc'])} |
| 3 s | baseline | {p(W[3]['split_acc'])} | {p(W[3]['grouped_cv_acc'])} |
| 5 s | baseline | {p(W[5]['split_acc'])} | {p(W[5]['grouped_cv_acc'])} |
| 10 s | baseline | {p(W[10]['split_acc'])} | {p(W[10]['grouped_cv_acc'])} |
| 5 s | `min, axis_std, jerk_raw, hf_frac, peaks_per_sec, step_period, spec_entropy, low_band_frac` | {p(R['window_final']['split_acc'])} | {p(ft['FINAL (8)']['dt_grouped_cv'])} (tree), {p(ft['FINAL (8)']['rf_grouped_cv'])} (random forest) |

""")
b, f = ft['baseline (7)'], ft['FINAL (8)']
answer(19, f"""
**Window length.** With the baseline features the depth-5 tree scores {p(W[2]['split_acc'])} / {p(W[3]['split_acc'])} / {p(W[5]['split_acc'])} / {p(W[10]['split_acc'])} at 2 / 3 / 5 / 10 s. That 3-point spread is about the noise of a single random split, and the grouped CV (whole recordings held out) is flat at 48-50%. So window length is not what limits this model; the features are. We locked in **5 s**: it holds about 9 steps at a normal cadence (enough for the autocorrelation and a 0.2 Hz FFT resolution), it keeps 612 of the 615 recordings (10 s windows drop every recording shorter than 8 s, leaving 603), and it gives a majority vote 5-10 windows per typical 40 s recording.

**Features.** At 5 s our 8 features raise the tree from {p(b['dt_split'])} to {p(f['dt_split'])} on the split and the random forest from {p(b['rf_grouped_cv'])} to {p(f['rf_grouped_cv'])} with recordings held out. The baseline features are all amplitude statistics of one signal, so the baseline tree never predicts the stairs classes (F1 0.00 for upstairs and downstairs). Rhythm features fixed that: `step_period` (autocorrelation lag) lifts downstairs F1 to {ft['baseline + step_period']['f1_downstairs']:.2f} and upstairs to {ft['baseline + step_period']['f1_upstairs']:.2f}; `spec_entropy` lifts downstairs to {ft['baseline + spec_entropy']['f1_downstairs']:.2f} and lying to {ft['baseline + spec_entropy']['f1_lying']:.2f}; `peaks_per_sec` and `zcr` separate jogging from walking (jogging F1 0.79 → 0.83). `axis_std` and `jerk_raw` measure 3-D motion without depending on how the phone is rotated, and `hf_frac` (power above 5 Hz in the unfiltered signal) separates standing (sway, 0.31) from sitting and lying (sensor noise, 0.50). We dropped `rms` (it duplicates `std`/`min`), `pca1` (no gain), `std_x/y/z` (they depend on how the phone sits in the pocket; a set with `std_z` transferred 1-2 points worse between devices in our tests), and `dom_freq`/`acf_peak`/`zcr` (overlap with `step_period` and `spec_entropy` once those are in).

**Still confused.** Upstairs vs walk (upstairs F1 only {f['f1_upstairs']:.2f} for the tree) and the three static classes (lying F1 {f['f1_lying']:.2f}). With gravity removed, the phone's orientation, which is what separates lying from sitting, is gone, and stairs differ from walking mostly along the vertical axis, which a magnitude signal can't isolate.
""")

# ---- Classification 1 (cell 22) ----
table(22, """| Default (max_depth=5, criterion='entropy') | **FILL UP** | **FILL UP** |
| **FILL UP best hyperparameters**  | **FILL UP** | **FILL UP** |""",
f"""| Default (max_depth=5, criterion='entropy') | {p(c1['default']['split'])} | {p(c1['default']['test'])} |
| {hp_dt(c1['best_params'])} | {p(c1['best']['split'])} | {p(c1['best']['test'])} |""")
f1d = c3['f1_dt']
answer(22, f"""
**What the scores say.** The tree does well on the held-out 30% of Sample_Data ({p(c1['best']['split'])}) but only {p(c1['best']['test'])} on the class corpus. The split score is optimistic: Sample_Data is 35 recordings of 3-5 min from 5 people, so windows from the same recording, and the same person, sit on both sides of the split and the tree is graded on people it has already seen. Class_Data is a new device (phone instead of wrist), new placements and about 100 new people, and the amplitudes don't line up: the wrist device's sensor noise is about 3x higher at rest (median `jerk_raw` 13-15 vs 4-8 for the phones in the three static classes), and wrist walking is gentler than pocket walking (median `axis_std` 2.5 vs 3.8 m/s², `min` -1.45 vs -2.38), so thresholds learned on the wrist land in the wrong place for phones.

**Which activities are distinguishable.** Jogging transfers best (F1 {f1d['jogging']:.2f}): it is 3-4x more intense than anything else on any device. Walking ({f1d['walk']:.2f}), standing ({f1d['standing']:.2f}) and sitting ({f1d['sitting']:.2f}) are middling, and the stairs (upstairs {f1d['upstairs']:.2f}, downstairs {f1d['downstairs']:.2f}) and lying ({f1d['lying']:.2f}) are close to chance.

**Hyperparameter tuning.** The grid prefers deeper trees (split accuracy rises from {p(c1['default']['split'])} at depth 5 to {p(c1['best']['split'])} at {hp_dt(c1['best_params'])}), but the class-corpus accuracy moves only {100 * (c1['best']['test'] - c1['default']['test']):+.1f} points. Tuning on a leaky split of five people makes the tree fit those five people better, not the class. The bottleneck is the domain shift, not the hyperparameters.
""")

# ---- Classification 2 (cell 25) ----
table(25, """| Default (max_depth=5, criterion='entropy') | **FILL UP** | **FILL UP** |
| **FILL UP best hyperparameters**  | **FILL UP** | **FILL UP** |""",
f"""| Default (max_depth=5, criterion='entropy') | {p(c2['default']['split'])} | {p(c2['default']['test'])} |
| {hp_dt(c2['best_params'])} | {p(c2['best']['split'])} | {p(c2['best']['test'])} |""")
f1d = c4['f1_dt']
answer(25, f"""
**What the scores say.** Trained on the class corpus, the tree reaches {p(c2['best']['split'])} on its own test split and {p(c2['best']['test'])} on Sample_Data. That is much better transfer than the other direction ({p(c1['best']['test'])}): 612 recordings from many phones, placements and people cover far more variation than 35 recordings from 5 wrists, so the class-trained tree has seen something like the wrist data before. The in-domain split is low (~55-59%) because Class_Data is noisy: recordings range from 4 s to 27 min, phones sit in pockets, hands or on tables, and some labels are loose (a "walk" file named "cooking").

**Which activities are distinguishable.** On Sample_Data, jogging is nearly perfect (F1 {f1d['jogging']:.2f}), and walking ({f1d['walk']:.2f}), standing ({f1d['standing']:.2f}), sitting ({f1d['sitting']:.2f}) and downstairs ({f1d['downstairs']:.2f}) are partly right. Upstairs ({f1d['upstairs']:.2f}) and lying ({f1d['lying']:.2f}) essentially fail: upstairs goes to walk/downstairs and lying to sitting/standing.

**Hyperparameter tuning.** The split picks {hp_dt(c2['best_params'])} ({p(c2['default']['split'])} → {p(c2['best']['split'])}), yet Sample_Data accuracy drops from {p(c2['default']['test'])} to {p(c2['best']['test'])}. The random split leaks windows of the same recording into the test set, so it rewards deeper trees that memorise recordings. A recording-grouped CV on Class_Data agrees with the transfer result: depth 5 gets 50.7% vs 49.6% for depth 8 (Model Exploration cell). Lesson: tune on a split that holds out whole recordings (or people).
""")

# ---- Classification 3 (cell 28) ----
table(28, """| **FILL UP best hyperparameters**  | **FILL UP** | **FILL UP** |""",
      f"""| {hp_rf(c3['best_params'])} | {p(c3['best']['split'])} | {p(c3['best']['test'])} |""")
gains = {c: c3['f1_rf'][c] - c3['f1_dt'][c] for c in c3['f1_dt']}
answer(28, f"""
**Random forest vs decision tree (Sample_Data → Class_Data).** The forest beats the best tree on both the Sample_Data split ({p(c3['best']['split'])} vs {p(c1['best']['split'])}) and the class corpus ({p(c3['best']['test'])} vs {p(c1['best']['test'])}, {100 * (c3['best']['test'] - c1['best']['test']):+.1f} points). Averaging 100 trees, each grown on a bootstrap sample with random feature subsets, smooths out the axis-aligned thresholds a single tree overfits to the five training people.

**Where the gain is.** It is uneven. Walk ({gains['walk']:+.2f} F1), upstairs ({gains['upstairs']:+.2f}), lying ({gains['lying']:+.2f}) and downstairs ({gains['downstairs']:+.2f}) improve; jogging is unchanged ({gains['jogging']:+.2f}) and standing/sitting get slightly worse ({gains['standing']:+.2f}/{gains['sitting']:+.2f}). The forest can't fix the device gap: absolute 46-48% is still far below what the same features reach when trained on the class corpus. Grid: depth 10 beats depth 5 (0.80 → 0.84 on the split); more trees and larger leaves change ≤1 point.
""")

# ---- Classification 4 (cell 31) ----
table(31, """| **FILL UP best hyperparameters**  | **FILL UP** | **FILL UP** |""",
      f"""| {hp_rf(c4['best_params'])} | {p(c4['best']['split'])} | {p(c4['best']['test'])} |""")
g4 = {c: c4['f1_rf'][c] - c4['f1_dt'][c] for c in c4['f1_dt']}
answer(31, f"""
**Random forest vs decision tree (Class_Data → Sample_Data).** The forest wins again: {p(c4['best']['split'])} vs {p(c2['best']['split'])} on the class split and {p(c4['best']['test'])} vs {p(c2['best']['test'])} on Sample_Data. Its most important features are `min` (0.23), `axis_std` (0.18) and `jerk_raw` (0.13), i.e. intensity first, then rhythm (`step_period` 0.11).

**Did it help all classes equally? No.** The gain is concentrated on the classes the tree almost never got right: upstairs F1 {c4['f1_dt']['upstairs']:.2f} → {c4['f1_rf']['upstairs']:.2f} ({g4['upstairs']:+.2f}) and lying {c4['f1_dt']['lying']:.2f} → {c4['f1_rf']['lying']:.2f} ({g4['lying']:+.2f}), then walk ({g4['walk']:+.2f}) and downstairs ({g4['downstairs']:+.2f}). Jogging was already solved ({c4['f1_rf']['jogging']:.2f}, {g4['jogging']:+.2f}). Upstairs and lying need combinations of several weak cues (slower rhythm plus lower jerk; noise level plus low-band share) that a depth-8 tree spends too few splits on; the forest's many trees cover those small regions. Even so, both stay below 0.3 F1, so most of the remaining error is in the features, not the model.
""")

# ---- Classification 5 (cell 38) ----
r = {(x['model'].split()[0], x['direction'].split()[0]): x for x in rec}
table(38, """| Decision Tree (best) | Class_Data → Sample_Data | **FILL UP** | **FILL UP** |
| **FILL UP your new classifier** (best) | Class_Data → Sample_Data | **FILL UP** | **FILL UP** |
| Decision Tree (best) | Sample_Data → Class_Data | **FILL UP** | **FILL UP** |
| **FILL UP your new classifier** (best) | Sample_Data → Class_Data | **FILL UP** | **FILL UP** |""",
f"""| Decision Tree (best) | Class_Data → Sample_Data | {p(r['Decision', 'Class_Data']['window_acc'])} | {p(r['Decision', 'Class_Data']['recording_acc'])} |
| Random Forest (best) | Class_Data → Sample_Data | {p(r['Random', 'Class_Data']['window_acc'])} | {p(r['Random', 'Class_Data']['recording_acc'])} |
| Decision Tree (best) | Sample_Data → Class_Data | {p(r['Decision', 'Sample_Data']['window_acc'])} | {p(r['Decision', 'Sample_Data']['recording_acc'])} |
| Random Forest (best) | Sample_Data → Class_Data | {p(r['Random', 'Sample_Data']['window_acc'])} | {p(r['Random', 'Sample_Data']['recording_acc'])} |""")
rf_cs = r['Random', 'Class_Data']
answer(38, f"""
**Does aggregating help?** Only a little overall: +{100 * (r['Decision', 'Class_Data']['recording_acc'] - r['Decision', 'Class_Data']['window_acc']):.1f} points for the tree and +{100 * (rf_cs['recording_acc'] - rf_cs['window_acc']):.1f} for the forest on Sample_Data, and it *loses* 2-3 points on Class_Data. A majority vote amplifies whatever the window classifier does on a recording. When most windows are right, the stray errors get outvoted: for the forest on Sample_Data, sitting goes {rf_cs['per_class_window']['sitting']:.0%} → {rf_cs['per_class_recording']['sitting']:.0%} and downstairs {rf_cs['per_class_window']['downstairs']:.0%} → {rf_cs['per_class_recording']['downstairs']:.0%}. When most windows are wrong, the vote makes the error total: lying ({rf_cs['per_class_window']['lying']:.0%} of windows right) and upstairs ({rf_cs['per_class_window']['upstairs']:.0%}) drop to 0% of recordings.

**When majority vote makes things worse.** (1) A class whose windows are right less than half the time, or whose errors all go to one wrong class: the vote turns a partial score into zero. (2) Short recordings: a 10 s recording has 1-2 windows, so one bad window decides it. (3) Window accuracy is dominated by a few very long recordings (one 27-minute sitting file alone gives 323 windows), while recording accuracy counts every recording once. That is why Class_Data drops at recording level: the short, messy recordings that rarely matter per window now count as much as the long clean ones. (4) Recordings that mix activities (e.g. phone handling at the start and end) can be outvoted toward the wrong label. We also tried confidence weighting (summing `predict_proba`): {p(r['Decision', 'Class_Data']['confidence_vote_acc'])} vs {p(r['Decision', 'Class_Data']['recording_acc'])} for the tree on Sample_Data, but {p(rf_cs['confidence_vote_acc'])} vs {p(rf_cs['recording_acc'])} for the forest, so we kept the majority vote and use summed probability only to break ties.
""")

nbformat.write(nb, 'A2_activity_recognition.ipynb')
print('filled')
