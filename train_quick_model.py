import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import dill

def main():
    # prefer augmented training file if present
    import os
    train_path = 'artifacts/train_aug.csv' if os.path.exists('artifacts/train_aug.csv') else 'artifacts/train.csv'
    df = pd.read_csv(train_path)
    if 'target' not in df.columns:
        raise RuntimeError('No target column in train.csv')

    X = df.drop(columns=['target'])
    y = df['target']

    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)

    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    X_train_p = preprocessor.fit_transform(X_train)

    # Option: use SMOTE oversampling if available
    use_smote = True
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X_train_p, y_train)
            X_train_fit = X_res
            y_train_fit = y_res
            print('Used SMOTE oversampling: new class counts ->', dict(pd.Series(y_train_fit).value_counts()))
        except Exception as e:
            print('SMOTE failed (trying RandomOverSampler):', e)
            try:
                from imblearn.over_sampling import RandomOverSampler
                ros = RandomOverSampler(random_state=42)
                X_res, y_res = ros.fit_resample(X_train_p, y_train)
                X_train_fit = X_res
                y_train_fit = y_res
                print('Used RandomOverSampler: new class counts ->', dict(pd.Series(y_train_fit).value_counts()))
            except Exception as e2:
                print('RandomOverSampler also failed, falling back to original data:', e2)
                X_train_fit = X_train_p
                y_train_fit = y_train
    else:
        X_train_fit = X_train_p
        y_train_fit = y_train

    # use class_weight as a fallback; increase estimators for stability
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    model.fit(X_train_fit, y_train_fit)

    # Save artifacts
    with open('artifacts/preprocessor.pkl', 'wb') as f:
        dill.dump(preprocessor, f)
    with open('artifacts/model.pkl', 'wb') as f:
        dill.dump(model, f)

    # quick eval
    X_val_p = preprocessor.transform(X_val)
    val_preds = model.predict(X_val_p)
    acc = (val_preds == y_val.values).mean()
    print(f'Trained RandomForest, val accuracy: {acc:.3f}')

if __name__ == '__main__':
    main()
