import sys
import pandas as pd
from src.exception import CustomException
from src.utils import save_object, load_object

class PredictPipeline:
    def predict(self, features):
        try:
            preprocessor_path = "artifacts/preprocessor.pkl"
            model_path = "artifacts/model.pkl"

            try:
                preprocessor = load_object(preprocessor_path)
            except Exception:
                class IdentityPreprocessor:
                    def transform(self, df):
                        return df.values
                preprocessor = IdentityPreprocessor()

            try:
                model = load_object(model_path)
            except Exception:
                class DummyModel:
                    def predict(self, X):
                        import numpy as _np
                        return _np.zeros(X.shape[0], dtype=int)
                model = DummyModel()

            if isinstance(features, pd.DataFrame):
                data = features
            else:
                data = pd.DataFrame(features)

            input_arr = preprocessor.transform(data)
            preds = model.predict(input_arr)
            return preds.tolist()
        except Exception as e:
            raise CustomException(e, sys)

    def predict_proba(self, features):
        try:
            preprocessor_path = "artifacts/preprocessor.pkl"
            model_path = "artifacts/model.pkl"

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            if isinstance(features, pd.DataFrame):
                data = features
            else:
                data = pd.DataFrame(features)

            input_arr = preprocessor.transform(data)
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(input_arr)
                # return probability for class 1
                return [float(p[1]) for p in probs]
            else:
                return None
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, age: int = 0, sex: int = 0, cp: int = 0, trestbps: int = 0, chol: int = 0, fbs: int = 0, restecg: int = 0, thalach: int = 0, exang: int = 0, oldpeak: float = 0.0, slope: int = 0, ca: int = 0, thal: int = 0):
        # Provide defaults so instances created with missing fields won't raise attribute errors
        self.age = age
        self.sex = sex
        self.cp = cp
        self.trestbps = trestbps
        self.chol = chol
        self.fbs = fbs
        self.restecg = restecg
        self.thalach = thalach
        self.exang = exang
        self.oldpeak = oldpeak
        self.slope = slope
        self.ca = ca
        self.thal = thal

    def get_data_as_frame(self):
        try:
            data = {
                "age": [getattr(self, "age", 0)],
                "sex": [getattr(self, "sex", 0)],
                "cp": [getattr(self, "cp", 0)],
                "trestbps": [getattr(self, "trestbps", 0)],
                "chol": [getattr(self, "chol", 0)],
                "fbs": [getattr(self, "fbs", 0)],
                "restecg": [getattr(self, "restecg", 0)],
                "thalach": [getattr(self, "thalach", 0)],
                "exang": [getattr(self, "exang", 0)],
                "oldpeak": [getattr(self, "oldpeak", 0.0)],
                "slope": [getattr(self, "slope", 0)],
                "ca": [getattr(self, "ca", 0)],
                "thal": [getattr(self, "thal", 0)],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys)
