import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load dataset
df = pd.read_csv("landslide_dataset.csv")

# Step 2: Fill missing values
df.fillna(df.select_dtypes(include='number').mean(), inplace=True)

# Step 3: Encode categorical columns
categorical_cols = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=categorical_cols)

# Step 4: Split features and target
# Split features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Predictions
predictions = model.predict(X_test)

# Step 8: Evaluation
print("MSE:", mean_squared_error(y_test, predictions))
print("R2:", r2_score(y_test, predictions))

# Step 9: Save model
joblib.dump(model, "landslide_model.pkl")

print("Model saved successfully")

