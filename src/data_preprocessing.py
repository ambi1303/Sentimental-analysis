import argparse
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

def generate_synthetic(path: str):
    np.random.seed(0)
    dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
    rows_per_day = 10
    data = []
    for date in dates:
        for _ in range(rows_per_day):
            q1 = np.random.randint(1, 6)
            q2 = np.random.randint(1, 6)
            recommend = np.random.choice(['Yes', 'No'])
            platform = np.random.choice(['Web', 'Mobile', 'In-Store'])
            avg = (q1 + q2) / 2
            if avg >= 4:
                label = 'Positive'
            elif avg <= 2:
                label = 'Negative'
            else:
                label = 'Neutral'
            data.append([date, q1, q2, recommend, platform, label])
    df = pd.DataFrame(data, columns=['date','Rating1','Rating2','Recommend','Platform','sentiment'])
    df.to_csv(path, index=False)
    print(f"Synthetic data saved to {path}")

def build_preprocessor():
    numeric = ['Rating1', 'Rating2']
    cat = ['Recommend', 'Platform']
    num_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])
    return ColumnTransformer([
        ('num', num_pipe, numeric),
        ('cat', cat_pipe, cat)
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    if args.generate:
        generate_synthetic('data/survey.csv')
    else:
        preprocessor = build_preprocessor()
        joblib.dump(preprocessor, 'data/preprocessor.joblib')
        print("Preprocessor saved.")

if __name__ == '__main__':
    main()
