import pandas as pd
import zipfile
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
zip_path = "archive.zip"  
extract_path = "energy_data"

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)
file_path = os.path.join(extract_path, "installed-solar-PV-capacity.csv")
df = pd.read_csv(file_path)

print("First 5 rows of dataset:")
print(df.head(), "\n")

print("Missing values in dataset:")
print(df.isnull().sum(), "\n")

df = df.dropna(subset=["Solar Capacity"])

if "Code" in df.columns:
    df = df.drop("Code", axis=1)

X = df.drop("Solar Capacity", axis=1)
y = df["Solar Capacity"]


if "Entity" in X.columns:
    le = LabelEncoder()
    X["Entity"] = le.fit_transform(X["Entity"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
sample = [[0, 2025]]  
prediction = model.predict(sample)
print("\nPredicted Solar Capacity for sample (Entity=0, Year=2025):", prediction[0])

