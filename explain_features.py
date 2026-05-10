import dill
import pandas as pd

def main():
    try:
        with open('artifacts/model.pkl','rb') as f:
            model = dill.load(f)
    except Exception as e:
        print('Failed to load model:', e); return

    try:
        with open('artifacts/preprocessor.pkl','rb') as f:
            pre = dill.load(f)
    except Exception as e:
        print('Failed to load preprocessor:', e); pre = None

    df = pd.read_csv('artifacts/train.csv')
    if 'target' not in df.columns:
        print('No target in train.csv'); return

    X = df.drop(columns=['target'])
    y = df['target']

    print('\nTop feature importances (model):')
    if hasattr(model, 'feature_importances_'):
        fi = model.feature_importances_
        cols = X.columns.tolist()
        imp = sorted(zip(cols, fi), key=lambda x: x[1], reverse=True)
        for c,v in imp[:10]:
            print(f'{c}: {v:.4f}')
    else:
        print('Model has no feature_importances_')

    print('\nMean values by target (positive vs negative) and diff (pos-neg):')
    means = df.groupby('target').mean()
    if 1 in means.index and 0 in means.index:
        for col in X.columns:
            pos = means.loc[1, col]
            neg = means.loc[0, col]
            diff = pos - neg
            direction = 'higher in positive' if diff>0 else ('lower in positive' if diff<0 else 'no diff')
            print(f'{col}: pos={pos:.3f}, neg={neg:.3f}, diff={diff:.3f} -> {direction}')
    else:
        print('Not enough classes to compute means')

    print('\nQuick guidance:')
    print('- Features with higher mean in positive cases and high importance generally increase risk when larger.')
    print('- For binary features (0/1), value 1 being higher in positive suggests presence increases risk.')

if __name__=='__main__':
    main()
