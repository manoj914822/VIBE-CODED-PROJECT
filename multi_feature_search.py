from src.utils import load_object
import pandas as pd
import numpy as np

# Greedy multi-feature search: at each step try changing each numeric feature across a grid,
# pick the single change that increases predicted probability the most, apply it, and repeat
# until probability >= threshold or max_steps reached.

def load_artifacts():
    model = load_object('artifacts/model.pkl')
    pre = load_object('artifacts/preprocessor.pkl')
    df_train = pd.read_csv('artifacts/train.csv')
    return model, pre, df_train


def get_baseline(df_train):
    row = {}
    for c in df_train.drop(columns=['target'], errors='ignore').columns:
        row[c] = float(df_train[c].mean())
    return pd.DataFrame([row])


def greedy_search(patient_df, model, pre, df_train, threshold=0.5, max_steps=5):
    current = patient_df.copy()
    feats = current.columns.tolist()
    numeric = current.select_dtypes(include=[float, int]).columns.tolist()
    Xt = pre.transform(current)
    if hasattr(model, 'predict_proba'):
        prob = float(model.predict_proba(Xt)[:,1][0])
    else:
        prob = float(model.predict(Xt)[0])
    history = []
    steps = 0
    while prob < threshold and steps < max_steps:
        best = None
        best_inc = 0.0
        best_val = None
        best_feat = None
        for col in numeric:
            mean = float(df_train[col].mean())
            std = float(df_train[col].std()) if df_train[col].std() > 0 else 1.0
            grid = list(np.linspace(mean - 3*std, mean + 3*std, 13)) + [mean*0.5, mean*1.5]
            grid = sorted(set([float(round(x,3)) for x in grid]))
            for val in grid:
                cand = current.copy()
                cand[col] = val
                try:
                    Xc = pre.transform(cand)
                    if hasattr(model, 'predict_proba'):
                        p = float(model.predict_proba(Xc)[:,1][0])
                    else:
                        p = float(model.predict(Xc)[0])
                except Exception:
                    continue
                inc = p - prob
                if inc > best_inc:
                    best_inc = inc
                    best = cand
                    best_val = val
                    best_feat = col
        if best is None:
            break
        # apply best
        current = best
        prob += best_inc
        history.append({'step': steps+1, 'feature': best_feat, 'value': best_val, 'prob': prob})
        steps += 1
    return prob, current, history


if __name__ == '__main__':
    model, pre, df_train = load_artifacts()
    patient = get_baseline(df_train)
    if 'trestbps' in patient.columns:
        patient['trestbps'] = 240.0
    print('Starting greedy multi-feature search for patient (trestbps=240)')
    prob0, cur0, hist0 = greedy_search(patient, model, pre, df_train, threshold=0.5, max_steps=6)
    print(f'Final prob after search: {prob0:.3f}')
    print('History:')
    for h in hist0:
        print(h)
    print('\nFinal feature values:')
    print(cur0.to_dict(orient='records')[0])
