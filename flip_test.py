from src.utils import load_object
import pandas as pd
import numpy as np

# Flip-test: vary each numerical feature across a grid and report first flip to positive

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


def flip_test_for_patient(patient_df, model, pre, df_train, threshold=0.5):
    numeric_cols = patient_df.select_dtypes(include=[float, int]).columns.tolist()
    baseline = patient_df.copy()
    orig_p = None
    Xp = pre.transform(baseline)
    if hasattr(model, 'predict_proba'):
        orig_p = float(model.predict_proba(Xp)[:,1][0])
    else:
        orig_p = float(model.predict(Xp)[0])
    print('Original prob:', orig_p)

    flips = []
    for col in numeric_cols:
        mean = float(df_train[col].mean())
        std = float(df_train[col].std()) if df_train[col].std() > 0 else 1.0
        # grid from mean - 3*std to mean + 3*std plus some extremes
        grid = list(np.linspace(mean - 3*std, mean + 3*std, 9)) + [mean*0.5, mean*1.5]
        grid = sorted(set([float(round(x,3)) for x in grid]))
        for val in grid:
            test = baseline.copy()
            test[col] = val
            try:
                Xt = pre.transform(test)
                if hasattr(model, 'predict_proba'):
                    p = float(model.predict_proba(Xt)[:,1][0])
                else:
                    p = float(model.predict(Xt)[0])
                if p >= threshold:
                    flips.append({'feature': col, 'flip_value': val, 'prob': p})
                    print(f"Flip found: {col} -> {val} (prob={p:.3f})")
                    break
            except Exception as e:
                continue
    if not flips:
        print('No single-feature flip found with the tried ranges.')
    return flips


if __name__ == '__main__':
    model, pre, df_train = load_artifacts()
    patient = get_baseline(df_train)
    # set the high BP example
    if 'trestbps' in patient.columns:
        patient['trestbps'] = 240.0
    print('Running flip-test for patient with trestbps=240')
    flips = flip_test_for_patient(patient, model, pre, df_train, threshold=0.5)
    print('\nFlip-test results:', flips)
