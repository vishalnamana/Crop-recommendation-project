import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================
# 🌱 CROP RECOMMENDATION MODEL
# ==============================

# Load crop dataset
crop_path = os.path.join(BASE_DIR, 'data', 'crop_recommendation.csv')
crop_df = pd.read_csv(crop_path)

# Features & target
X_crop = crop_df.drop('label', axis=1)
y_crop = crop_df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_crop, y_crop, test_size=0.2, random_state=42
)

# Train model
crop_model = RandomForestClassifier()
crop_model.fit(X_train, y_train)

# Create model folder
os.makedirs(os.path.join(BASE_DIR, 'model'), exist_ok=True)

# Save crop model
pickle.dump(crop_model, open(os.path.join(BASE_DIR, 'model', 'crop_model.pkl'), 'wb'))

print("✅ Crop model trained and saved!")

# ==============================
# 🌾 YIELD PREDICTION MODEL
# ==============================

# Load yield dataset
yield_path = os.path.join(BASE_DIR, 'data', 'yield.csv')
yield_df = pd.read_csv(yield_path)

print("Yield Dataset Columns:", yield_df.columns)

# Remove unwanted column
if 'Unnamed: 0' in yield_df.columns:
    yield_df = yield_df.drop(['Unnamed: 0'], axis=1)

# Convert categorical (text) → numeric
yield_df = pd.get_dummies(yield_df)

# Target column
target_column = 'hg/ha_yield'

# Features & target
X_yield = yield_df.drop(target_column, axis=1)
y_yield = yield_df[target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_yield, y_yield, test_size=0.2, random_state=42
)

# Train model
yield_model = LinearRegression()
yield_model.fit(X_train, y_train)

# Save yield model
pickle.dump(yield_model, open(os.path.join(BASE_DIR, 'model', 'yield_model.pkl'), 'wb'))

print("✅ Yield model trained and saved!")