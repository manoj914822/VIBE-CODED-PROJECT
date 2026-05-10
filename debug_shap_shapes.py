from src.utils import load_object
import pandas as pd

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

print('Xbg type:', type(Xbg), 'shape:', getattr(Xbg,'shape',None), 'dtype:', getattr(Xbg,'dtype',None))
print('Xp type:', type(Xp), 'shape:', getattr(Xp,'shape',None), 'dtype:', getattr(Xp,'dtype',None))
print('First row Xbg sample (first 8):', Xbg.flatten()[:8])
print('Xp flatten (first 8):', Xp.flatten()[:8])
print('Model type:', type(model))
try:
    import numpy as np
    print('Xbg ndim:', Xbg.ndim)
    print('Xp ndim:', Xp.ndim)
except Exception as e:
    print('ndim check failed:', e)
