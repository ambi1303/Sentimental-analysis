import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load data and preprocessor
DF = pd.read_csv('data/survey.csv', parse_dates=['date'])
preprocessor = joblib.load('data/preprocessor.joblib')

# Features & label
y = DF['sentiment']
X = DF[['Rating1','Rating2','Recommend','Platform']]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42)

# Full pipeline
model = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train & evaluate
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))

# Save model
joblib.dump(model, 'data/sentiment_model.joblib')
print("Model saved.")
