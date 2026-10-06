"Predict one delivery, from the command line."

import pandas as pd
import joblib

def main():
    # load model
    model = joblib.load("model.joblib")

    # create input (7 km, 25 min prep, traffic 3, no rain)
    X = pd.DataFrame([{
        "distance_km": 7,
        "prep_time_min": 25,
        "traffic_level": 3,
        "rain": 0
    }])

    # predict
    pred = model.predict(X)[0]

    print(f"PREDICTION: {pred:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
