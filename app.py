import streamlit as st
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.utils import load_object

st.set_page_config(page_title="Heart Disease Predictor")
st.title("Heart Disease Prediction Dashboard")
st.write("Enter clinical parameters to predict heart disease risk.")

# Input fields
with st.form(key='input_form'):
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[0, 1], index=0)
    cp = st.selectbox("Chest pain type (cp)", options=[0,1,2,3], index=0)
    trestbps = st.number_input("Resting blood pressure (trestbps)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Serum cholestoral (chol)", min_value=50, max_value=600, value=200)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", options=[0,1], index=0)
    restecg = st.selectbox("Resting ECG (restecg)", options=[0,1,2], index=0)
    thalach = st.number_input("Max heart rate achieved (thalach)", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise induced angina (exang)", options=[0,1], index=0)
    oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of ST segment (slope)", options=[0,1,2], index=0)
    ca = st.selectbox("Number of major vessels (ca)", options=[0,1,2,3,4], index=0)
    thal = st.selectbox("Thalassemia (thal)", options=[0,1,2,3], index=0)

    submit = st.form_submit_button("Predict")

if submit:
    try:
        input_obj = CustomData(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs, restecg=restecg, thalach=thalach, exang=exang, oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)
        input_df = input_obj.get_data_as_frame()
        pipeline = PredictPipeline()
        preds = pipeline.predict(input_df)
        probs = None
        try:
            probs = pipeline.predict_proba(input_df)
        except Exception:
            probs = None

        result = preds[0] if preds else None
        if result is None:
            st.error("No prediction returned")
        else:
            prob = probs[0] if probs is not None else None
            # allow user to choose threshold
            thresh = st.sidebar.slider("Classification threshold (P>=)", 0.0, 1.0, 0.3, 0.01)
            if prob is not None:
                label = 'Yes' if prob >= thresh else 'No'
                st.success(f"Prediction: {label} — P(class=1)={prob:.3f} (threshold={thresh:.2f})")
            else:
                label = 'Yes' if int(result) == 1 else 'No'
                st.success(f"Prediction: {label} (raw={result})")

        # Model insights
        if st.button("Show model insights"):
            try:
                model = load_object('artifacts/model.pkl')
                preprocessor = load_object('artifacts/preprocessor.pkl')
                # show feature importances if available
                if hasattr(model, 'feature_importances_'):
                    fi = model.feature_importances_
                    cols = input_df.columns.tolist()
                    imp_df = pd.DataFrame({'feature': cols, 'importance': fi})
                    imp_df = imp_df.sort_values('importance', ascending=False)
                    st.write("Top features by importance:")
                    st.dataframe(imp_df.head(10))
                else:
                    st.write("Model has no feature_importances_")

                # show sample prediction distribution on train set if available
                try:
                    df_train = pd.read_csv('artifacts/train.csv')
                    X_sample = df_train.drop(columns=['target'], errors='ignore')
                    Xp = preprocessor.transform(X_sample)
                    if hasattr(model, 'predict_proba'):
                        probs_all = model.predict_proba(Xp)[:,1]
                        st.write('Train set P(class=1) mean:', float(probs_all.mean()))
                        st.write('Train set P(class=1) distribution (sample):')
                        st.write(pd.Series(probs_all).describe())
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Model insights error: {e}")
        # Explain this prediction (simple approximation + optional SHAP)
        if st.button("Explain prediction"):
            try:
                model = load_object('artifacts/model.pkl')
                df_train = pd.read_csv('artifacts/train.csv')
                feature_means = df_train.drop(columns=['target'], errors='ignore').mean()
                feature_list = input_df.columns.tolist()
                if hasattr(model, 'feature_importances_'):
                    fi = model.feature_importances_
                    contribs = []
                    for idx, col in enumerate(feature_list):
                        mean_val = feature_means.get(col, 0)
                        val = float(input_df.iloc[0][col])
                        contrib = fi[idx] * (val - mean_val)
                        contribs.append((col, contrib, val, mean_val))
                    contribs = sorted(contribs, key=lambda x: abs(x[1]), reverse=True)
                    rows = []
                    for col, contrib, val, mean_val in contribs[:8]:
                        direction = 'increases risk' if contrib > 0 else 'decreases risk' if contrib < 0 else 'no effect'
                        rows.append({'feature': col, 'value': val, 'mean': mean_val, 'contrib': float(contrib), 'direction': direction})
                    st.write('Top feature contributions (approx):')
                    st.dataframe(pd.DataFrame(rows))
                else:
                    st.write('Model does not expose feature importances for explanation')
            except Exception as e:
                st.error(f'Explain error: {e}')

        if st.button('Explain with SHAP'):
            try:
                import shap
                model_obj = load_object('artifacts/model.pkl')
                pre_obj = load_object('artifacts/preprocessor.pkl')
                Xp = pre_obj.transform(input_df)
                # Use TreeExplainer for tree models
                try:
                    explainer = shap.TreeExplainer(model_obj)
                    shap_values = explainer.shap_values(Xp)
                    # shap_values for binary classifier is list; take index 1
                    if isinstance(shap_values, list):
                        sv = shap_values[1]
                    else:
                        sv = shap_values
                    sv_row = sv[0]
                    feats = input_df.columns.tolist()
                    df_shap = pd.DataFrame({'feature': feats, 'shap_value': sv_row, 'value': input_df.iloc[0].values})
                    df_shap['abs_shap'] = df_shap['shap_value'].abs()
                    df_shap = df_shap.sort_values('abs_shap', ascending=False).head(8)
                    st.write('Top SHAP contributions:')
                    st.dataframe(df_shap[['feature','value','shap_value']])
                except Exception as e:
                    st.write('SHAP TreeExplainer failed:', e)
            except Exception as e:
                st.error('SHAP not available: run pip install shap')
            except Exception as e:
                st.error(f'Explain error: {e}')
    except Exception as e:
        st.error(f"Prediction error: {e}")
