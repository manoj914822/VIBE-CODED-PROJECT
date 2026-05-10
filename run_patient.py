from src.pipeline.predict_pipeline import PredictPipeline
import pandas as pd

def map_and_run():
    # Input from the user (Clinical Intelligence Workspace payload)
    patient = {
        "age": 50,
        # sex: 1=male, 0=female
        "sex": 1,
        # chest pain mapping: Typical Angina -> 0
        "cp": 0,
        "trestbps": 120,
        "chol": 200,
        # fasting blood sugar > 120 mg/dl -> assume No (0)
        "fbs": 0,
        # resting ECG Normal -> 0
        "restecg": 0,
        "thalach": 150,
        # Exercise induced angina No -> 0
        "exang": 0,
        "oldpeak": 1.0,
        # slope Flat -> map to 1
        "slope": 1,
        "ca": 0,
        # thal map default 1
        "thal": 1,
    }

    df = pd.DataFrame([patient])
    pp = PredictPipeline()
    preds = pp.predict(df)
    print("Patient input:")
    print(df.to_dict(orient='records')[0])
    label = 'Yes' if int(preds[0]) == 1 else 'No'
    print("Prediction:", label, "(raw=", preds[0], ")")

if __name__ == '__main__':
    map_and_run()
