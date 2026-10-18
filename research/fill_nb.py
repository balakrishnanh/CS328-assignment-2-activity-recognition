import sys, nbformat
sys.path.insert(0, 'research')
import cells as C
nb = nbformat.read('research/A2_original.ipynb', as_version=4)
cs = nb.cells
def code(src): return nbformat.v4.new_code_cell(src)
def md(src): return nbformat.v4.new_markdown_cell(src)
cs[2].source = C.IMPORTS
cs[5].source = C.EXTRACT
cs[7].source = C.COMBINE
cs[9].source = C.MYFEATS
cs[17].source = C.FEATURE_EXPLORATION
cs[20].source = C.CLASS1
cs[23].source = C.CLASS2
cs[26].source = C.CLASS3
cs[29].source = C.CLASS4
cs[32].source = C.PREDICT_REC
cs[34].source = C.CLASS5
cs[37].source = C.FAILURE
new = cs[:18] + [code(C.FEATURE_EXPLORATION2)] + cs[18:31] + [md('## Model Exploration: same features, nine classifiers (exploration report §2)\n\nEvery model uses the locked 5 s windows and `FEATURES`. The notebook split puts windows from the same recording on both sides of the split, so it overstates accuracy; the grouped CV holds out whole recordings and is the number we trust for choosing a model.'), code(C.MODEL_EXPLORATION)] + cs[31:]
nb.cells = new
for c in nb.cells:
    if c.cell_type == 'code': c.outputs = []; c.execution_count = None
nbformat.write(nb, 'A2_activity_recognition.ipynb')
print(len(nb.cells))
