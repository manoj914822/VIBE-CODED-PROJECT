from src.utils import load_object
import pandas as pd, traceback
model = load_object('artifacts/model.pkl')
pre = load_object('artifacts/preprocessor.pkl')
df_train = pd.read_csv('artifacts/train.csv')
bg_df = df_train.drop(columns=['target'], errors='ignore').sample(n=min(50, len(df_train)), random_state=42)
Xbg = pre.transform(bg_df)
row = {c: float(df_train[c].mean()) for c in df_train.drop(columns=['target'], errors='ignore').columns}
if 'trestbps' in row:
    row['trestbps'] = 240.0
input_df = pd.DataFrame([row])
Xp = pre.transform(input_df)

import shap
try:
    print('Calling TreeExplainer...')
    explainer = shap.TreeExplainer(model, data=Xbg)
    print('TreeExplainer created, calling shap_values...')
    shap_values = explainer.shap_values(Xp)
    print('shap_values computed')
    print(type(shap_values))
except Exception as e:
    print('Exception during TreeExplainer/shap_values:')
    traceback.print_exc()

try:
    print('Trying shap.Explainer with predict_proba...')
    expl2 = shap.Explainer(model.predict_proba, Xbg)
    out = expl2(Xp)
    print('Explainer returned, values shape:', getattr(out, 'values', None).shape)
except Exception:
    print('Exception during shap.Explainer:')
    traceback.print_exc()
