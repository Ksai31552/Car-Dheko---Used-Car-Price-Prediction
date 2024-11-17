# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 22:22:44 2024

@author: USER
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# path = "C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/combined_data.csv"


# filename = os.listdir(path)

# cdf = pd.DataFrame()
# for file in filename:
#     if file.endswith(".csv"):
#         print(file)
#         df = pd.read_csv(path+'/'+file, encoding='utf-8')
#         cdf = pd.concat([cdf,df], ignore_index=True)
        
# cdf['price'] = cdf['price'].str.replace('[^0-9a-zA-Z., ]+', '', regex=True)

# cdf['price'] = cdf['price'].str.strip()
# cdf['price'] = cdf['price'].fillna(0)
# cdf['price'] = cdf['price'].str.replace(",", "")
# cdf['price'] = cdf['price'].str.replace("Unknown",'0')

# cdf['amt_type'] = cdf['price'].str.split().str[1]
# cdf['price'] = cdf['price'].str.replace("Lakh", "").str.replace("Crore","")
# # ls = list(cdf.columns)

# cdf = cdf[['trendingText', 'trendingText2', 'trendingText3', 'it', 'ft', 'bt', 'km', 'transmission', 'ownerNo', 'owner', 'oem', 'model', 'modelYear', 'centralVariantId', 'variantName', 'price', 'amt_type', 'priceActual', 'priceSaving', 'priceFixedText', 'id', 'city', 'Registration Year', 'Insurance Validity', 'Fuel Type', 'Seats', 'Kms Driven', 'RTO', 'Ownership', 'Engine Displacement', 'Transmission', 'Year of Manufacture', 'heading', 'Power Steering', 'Power Windows Front', 'Air Conditioner', 'Heater', 'Adjustable Head Lights', 'Manually Adjustable Exterior Rear View Mirror', 'Centeral Locking', 'Child Safety Locks', 'heading.1', 'Fog Lights Front', 'Anti Lock Braking System', 'Cd Player', 'Radio', 'Power Adjustable Exterior Rear View Mirror', 'Brake Assist', 'Electric Folding Rear View Mirror', 'Power Door Locks', 'Cd Changer', 'Fog Lights Rear', 'Remote Trunk Opener', 'Tinted Glass', 'Halogen Headlamps', 'Usb Auxiliary Input', 'Passenger Side Rear View Mirror', 'Cassette Player', 'Day Night Rear View Mirror', 'Power Windows Rear', 'Leather Seats', 'Leather Steering Wheel', 'Rear Seat Belts', 'Remote Fuel Lid Opener', 'Bluetooth', 'Speakers Front', 'Power Antenna', 'Dvd Player', 'Audio System Remote Control', 'Wheel Covers', 'Rear Window Wiper', 'Driver Air Bag', 'Low Fuel Warning Light', 'Cup Holders Front', 'Navigation System', 'Tachometer', 'Glove Compartment', 'Integrated2Din Audio', 'Digital Odometer', 'Fabric Upholstery', 'Accessory Power Outlet', 'Rear Seat Headrest', 'Speakers Rear', 'Adjustable Steering', 'Anti Theft Alarm', 'Number Of Speaker', 'Mileage', 'Engine', 'Max Power', 'Torque', 'Seats.1', 'heading.2', 'Wheel Size', 'Air Quality Control', 'Electronic Multi Tripmeter', 'Trunk Light', 'Vanity Mirror', 'Rear Window Defogger', 'Adjustable Seats', 'Tyre Pressure Monitor', 'Touch Screen', 'Integrated Antenna', 'Multifunction Steering Wheel', 'Sun Roof', 'Moon Roof', 'Side Air Bag Front', 'Cruise Control', 'Drive Modes', 'Digital Clock', 'Passenger Air Bag', 'Rear Folding Table', 'Driving Experience Control Eco', 'Engine Immobilizer', 'Rear Spoiler', 'Alloy Wheels', 'LEDDRLs']]

# cdf['price'] = cdf['price'].astype(float)
# cdf['price'] = cdf['price'].fillna(cdf['price'].mean())

# cdf.to_csv("D:/Sai_vigneshwar/combined_data.csv")
df = pd.read_csv("C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/combined_data.csv")



def clean_numeric_column(column, df):
    # Remove non-numeric characters and convert to float
    return pd.to_numeric(df[column].str.replace(r'[^0-9.]', '', regex=True), errors='coerce')

numeric_columns = ['km', 'Kms Driven', 'Mileage']

for col in numeric_columns:
    df[col] = clean_numeric_column(col, df)
# df['Mileage'].head()

# Define a strategy for numerical and categorical columns
numerical_imputer = SimpleImputer(strategy='mean')  # Use 'median' or 'most_frequent' if needed
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Apply the imputer to numerical and categorical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns



df[numerical_columns] = numerical_imputer.fit_transform(df[numerical_columns])
df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

# OneHotEncode categorical variables
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_categorical = pd.DataFrame(encoder.fit_transform(df[categorical_columns]), 
                                   columns=encoder.get_feature_names_out(categorical_columns))

df = df.drop(categorical_columns, axis=1)
df = pd.concat([df, encoded_categorical], axis=1)

scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Step 5: Model Development

# Assuming 'price' is the target variable
X = df.drop('price', axis=1)  # Features
y = df['price']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor as an example model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')

r2 = r2_score(y_test, y_pred)
print(f'R-squared (R²) for acccuracy: {r2}')

# correlation_matrix = X.corr()

# print(correlation_matrix)


models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Support Vector Regressor": SVR(),
    "K-Nearest Neighbors": KNeighborsRegressor()
}

imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the training data and transform both training and test sets
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# X_train = X_train.dropna()
# X_train = X_train[~np.isnan(X_train).any(axis=1)]
# y_train = y_train[X_train.index]

# Drop rows with missing values from X_test and y_test
# X_test = X_test.dropna()
# y_test = y_test[X_test.index]

# Dictionary to store evaluation metrics for each model
model_metrics = {}

# Train, predict, and evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    
    # Store the metrics
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


####################################################################

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
