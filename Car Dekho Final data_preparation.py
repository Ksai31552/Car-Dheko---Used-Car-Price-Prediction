# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Reading the data
df = pd.read_csv("C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/combined_data.csv")

# Function to clean numeric columns
def clean_numeric_column(column, df):
    return pd.to_numeric(df[column].str.replace(r'[^0-9.]', '', regex=True), errors='coerce')

numeric_columns = ['km', 'Kms Driven', 'Mileage']

for col in numeric_columns:
    df[col] = clean_numeric_column(col, df)

# Imputing missing values for numerical and categorical columns
numerical_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

df[numerical_columns] = numerical_imputer.fit_transform(df[numerical_columns])
df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

# OneHotEncode categorical variables
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_categorical = pd.DataFrame(encoder.fit_transform(df[categorical_columns]), 
                                   columns=encoder.get_feature_names_out(categorical_columns))

df = df.drop(categorical_columns, axis=1)
df = pd.concat([df, encoded_categorical], axis=1)

# Standardize numerical columns
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Define features and target variable
X = df.drop('price', axis=1)  # Assuming 'price' is the target variable
y = df['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute any missing values (if needed) after split
imputer = SimpleImputer(strategy='mean')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Train and evaluate models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Support Vector Regressor": SVR(),
    "K-Nearest Neighbors": KNeighborsRegressor()
}

# Dictionary to store evaluation metrics
model_metrics = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    
    model_metrics[model_name] = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2
    }

# Display results
for model_name, metrics in model_metrics.items():
    print(f"Model: {model_name}")
    print(f"  Mean Absolute Error (MAE): {metrics['MAE']}")
    print(f"  Mean Squared Error (MSE): {metrics['MSE']}")
    print(f"  Root Mean Squared Error (RMSE): {metrics['RMSE']}")
    print(f"  R-squared (R²): {metrics['R²']}")
    print("\n")
