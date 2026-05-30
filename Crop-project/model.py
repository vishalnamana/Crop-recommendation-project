import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
data = pd.read_csv("data/Crop_recommendation.csv")

# Input and Output
X = data.drop("label", axis=1)
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save trained model
joblib.dump(model, "model/crop_model.pkl")

print("Model saved successfully as model/crop_model.pkl")