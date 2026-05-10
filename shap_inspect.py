from src.utils import load_object
import pandas as pd
import numpy as np

def main():
    model = load_object('artifacts/model.pkl')
    pre = load_object('artifacts/preprocessor.pkl')
    df_train = pd.read_csv('artifacts/train.csv')

    # baseline patient: training means
    row = {c: float(df_train[c].mean()) for c in df_train.drop(columns=['target'], errors='ignore').columns}
    if 'trestbps' in row:
        row['trestbps'] = 240.0
    input_df = pd.DataFrame([row])
    print('Input used:')
    print(input_df.to_dict(orient='records')[0])

    Xp = pre.transform(input_df)

    try:
        import shap
    except Exception as e:
        print('shap not installed:', e)
        return

    try:
        # build background from train samples (up to 50 rows)
        bg_df = df_train.drop(columns=['target'], errors='ignore').sample(n=min(50, len(df_train)), random_state=42)
        Xbg = pre.transform(bg_df)
        # Try TreeExplainer with a background dataset first
        try:
            explainer = shap.TreeExplainer(model, data=Xbg)
            shap_values = explainer.shap_values(Xp)
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            else:
                sv = shap_values[0]
        except Exception:
            # fallback to generic Explainer over predict_proba
            explainer = shap.Explainer(model.predict_proba, Xbg)
            ev = explainer(Xp)
            vals = ev.values
            if vals.ndim == 3:
                sv = vals[0,:,1]
            else:
                sv = vals[0,:]

        feats = input_df.columns.tolist()
        print('DEBUG: feats len', len(feats))
        print('DEBUG: input values shape', input_df.iloc[0].values.shape)
        import numpy as _np
        print('DEBUG: sv type, shape:', type(sv), getattr(_np.array(sv),'shape',None))
        # normalize sv to 1D array of length n_features
        sv_arr = _np.array(sv)
        if sv_arr.ndim == 0:
            sv_flat = _np.array([float(sv_arr)])
        elif sv_arr.ndim == 1:
            sv_flat = sv_arr
        elif sv_arr.ndim == 2:
            # try to pick the row or column which matches feature count
            if sv_arr.shape[0] == 1:
                sv_flat = sv_arr[0]
            elif sv_arr.shape[1] == len(feats):
                sv_flat = sv_arr[:, -1] if sv_arr.shape[0] != len(feats) else sv_arr[:,0]
            else:
                # fallback: flatten and take first n features
                sv_flat = sv_arr.flatten()[:len(feats)]
        else:
            sv_flat = sv_arr.flatten()[:len(feats)]

        df_shap = pd.DataFrame({'feature': feats, 'value': input_df.iloc[0].values, 'shap_value': sv_flat})
        df_shap['abs_shap'] = df_shap['shap_value'].abs()
        df_shap = df_shap.sort_values('abs_shap', ascending=False).head(10)
        print('\nTop SHAP contributions:')
        print(df_shap[['feature','value','shap_value']].to_string(index=False))
    except Exception as e:
        import traceback
        print('SHAP explanation failed:', e)
        traceback.print_exc()

if __name__ == '__main__':
    main()
