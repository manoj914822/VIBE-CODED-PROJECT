from src.utils import load_object
import pandas as pd
import numpy as np

def main():
    print('Loading artifacts...')
    try:
        model = load_object('artifacts/model.pkl')
    except Exception as e:
        print('Failed to load model:', e)
        return
    try:
        pre = load_object('artifacts/preprocessor.pkl')
    except Exception as e:
        print('Failed to load preprocessor:', e)
        pre = None

    df_train = None
    try:
        df_train = pd.read_csv('artifacts/train.csv')
    except Exception as e:
        print('Failed to read artifacts/train.csv:', e)

    if df_train is None:
        print('No train.csv available — cannot infer feature order. Exiting.')
        return

    cols = df_train.drop(columns=['target'], errors='ignore').columns.tolist()
    # build a baseline row using training means
    row = {}
    for c in cols:
        if c in df_train.columns:
            row[c] = float(df_train[c].mean())
        else:
            row[c] = 0.0

    # set a very high resting blood pressure
    if 'trestbps' in row:
        row['trestbps'] = 240.0

    input_df = pd.DataFrame([row])
    print('\nInput values (used for prediction):')
    print(input_df.to_dict(orient='records')[0])

    Xp = None
    if pre is not None:
        try:
            Xp = pre.transform(input_df)
            print('\nPreprocessor transform output shape:', Xp.shape)
            # show first 8 transformed values
            print('Preprocessed vector (first 8 vals):', Xp.flatten()[:8])
        except Exception as e:
            print('Preprocessor.transform failed:', e)

    try:
        if Xp is None:
            # try direct model.predict on raw data if model accepts it
            pred = model.predict(input_df)
            print('\nModel predicted (raw input):', pred)
            if hasattr(model, 'predict_proba'):
                p = model.predict_proba(input_df)[:,1]
                print('Predicted prob (raw input):', float(p[0]))
        else:
            pred = model.predict(Xp)
            print('\nModel predicted (after preprocess):', int(pred[0]))
            if hasattr(model, 'predict_proba'):
                p = model.predict_proba(Xp)[:,1]
                print('Predicted prob (after preprocess):', float(p[0]))
    except Exception as e:
        print('Model prediction failed:', e)

    # If model has feature_importances_, show contribution approximation
    try:
        if hasattr(model, 'feature_importances_') and pre is not None:
            fi = model.feature_importances_
            # try to map back using preprocessor input feature order == cols
            contribs = []
            for i, col in enumerate(cols):
                val = input_df.iloc[0][col]
                mean = float(df_train[col].mean())
                contrib = fi[i] * (val - mean)
                contribs.append((col, float(contrib), float(val), float(mean)))
            contribs = sorted(contribs, key=lambda x: abs(x[1]), reverse=True)[:8]
            print('\nTop approx feature contributions:')
            for c, contrib, val, mean in contribs:
                print(f"{c}: contrib={contrib:.4f}, val={val}, mean={mean}")
    except Exception:
        pass

if __name__ == '__main__':
    main()
