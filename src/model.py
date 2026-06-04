import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Generate Data
np.random.seed(42)
n_samples = 1000
data = {
    'Age': np.random.randint(18, 70, size=n_samples),
    'Tenure': np.random.randint(1, 72, size=n_samples),
    'MonthlyCharges': np.random.round(np.random.uniform(20, 120, size=n_samples), 2),
    'TotalCharges': np.random.uniform(100, 5000, size=n_samples),
    'Contract': np.random.choice([0, 1, 2], size=n_samples),
    'TechSupport': np.random.choice([0, 1], size=n_samples),
    'Churn': np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25])
}
df = pd.DataFrame(data)

# Train Model
X = df[['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'TechSupport']]
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save
joblib.dump(model, 'src/churn_model.pkl')
print("✅ Model trained and saved successfully inside src folder!")
