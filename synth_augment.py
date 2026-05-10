import pandas as pd
import numpy as np

# Simple synthetic augmentation: take positive class examples and add gaussian noise
# to numeric features to create N synthetic positive samples and save to artifacts/train_aug.csv

def main(n=50, sigma=0.05):
    df = pd.read_csv('artifacts/train.csv')
    if 'target' not in df.columns:
        print('No target column in train.csv')
        return
    pos = df[df['target']==1]
    if len(pos) == 0:
        print('No positive samples to augment')
        return
    numeric_cols = pos.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c!='target']
    synth = []
    rng = np.random.default_rng(42)
    for i in range(n):
        base = pos.sample(1, replace=True).iloc[0]
        new = base.copy()
        for c in numeric_cols:
            val = float(base[c])
            noise = rng.normal(0, sigma*max(1.0, abs(val)))
            new[c] = max(0, val + noise)
        new['target'] = 1
        synth.append(new)
    df_aug = pd.concat([df, pd.DataFrame(synth)], ignore_index=True)
    df_aug.to_csv('artifacts/train_aug.csv', index=False)
    print(f'Wrote artifacts/train_aug.csv with {len(df_aug)} rows (added {n} synthetic positives)')

if __name__ == '__main__':
    main(n=100, sigma=0.1)
