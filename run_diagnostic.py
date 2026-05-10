import sys
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline
from src.exception import CustomException

def run():
    try:
        test_path = "artifacts/test.csv"
        df = pd.read_csv(test_path)

        # If label column exists, drop it for prediction
        X = df.drop(columns=[c for c in ["target", "label"] if c in df.columns], errors="ignore")

        pp = PredictPipeline()
        preds = pp.predict(X)

        print(f"Ran diagnostic on {len(X)} rows. Sample predictions:\n{preds[:10]}")
        # print distribution
        from collections import Counter
        print("Prediction distribution:", Counter(preds))
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    run()
