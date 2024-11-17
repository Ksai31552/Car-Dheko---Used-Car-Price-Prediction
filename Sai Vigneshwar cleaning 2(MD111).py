# -*- coding: utf-8 -*-
"""


@author: Sai.Vigneshwar
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy import stats
import json
import ast
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define file paths
chennai_file_path = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/Save files/chennai_cars.csv'
bangalore_file_path = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/Save files/bangalore_cars.csv'
delhi_file_path = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/Save files/delhi_cars.csv'
hyderabad_file_path = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/Save files/hyderabad_cars.csv'
jaipur_file_path = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/Save files/jaipur_cars.csv'
kolkata_file_path = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/Save files/kolkata_cars.csv'

# If the above encoding doesn't work, try 'utf-8'
chennai_df = pd.read_csv(chennai_file_path, encoding='utf-8')
bangalore_df = pd.read_csv(bangalore_file_path, encoding='utf-8')
delhi_df = pd.read_csv(delhi_file_path, encoding='utf-8')
hyderabad_df = pd.read_csv(hyderabad_file_path, encoding='utf-8')
jaipur_df = pd.read_csv(jaipur_file_path, encoding='utf-8')
kolkata_df = pd.read_csv(kolkata_file_path, encoding='utf-8')


# Print column names to debug
print("Chennai DataFrame columns:", chennai_df.columns)
print("Bangalore DataFrame columns:",bangalore_df.columns)
print("Delhi DataFrame columns:", delhi_df.columns)
print("Hyderabad DataFrame columns:", hyderabad_df.columns)
print("Jaipur DataFrame columns:", jaipur_df.columns)
print("Kolkata DataFrame columns:", kolkata_df.columns)


# Selecting only the columns want to keep
selected_columns = [
    'price', 'Registration Year', 'city', 'Power Steering', 'Anti Lock Braking System', 
    'bt', 'km', 'ownerNo', 'oem', 'model', 'centralVariantId', 'variantName', 
    'Insurance Validity', 'Fuel Type', 'Seats', 'Engine Displacement', 
    'Transmission', 'Year of Manufacture', 'Max Power', 'Mileage'
]

# Filter the DataFrame to include only the selected columns
chennai_df = chennai_df[selected_columns]
bangalore_df = bangalore_df[selected_columns]
delhi_df = delhi_df[selected_columns]
hyderabad_df = hyderabad_df[selected_columns]
jaipur_df = jaipur_df[selected_columns]
kolkata_df = kolkata_df[selected_columns]

# If the encoding artifacts persist, remove them using a broader replace method
chennai_df['price'] = chennai_df['price'].str.replace('[^0-9a-zA-Z., ]+', '', regex=True)
bangalore_df['price'] = chennai_df['price'].str.replace('[^0-9a-zA-Z., ]+', '', regex=True)
delhi_df['price'] = chennai_df['price'].str.replace('[^0-9a-zA-Z., ]+', '', regex=True)
hyderabad_df['price'] = chennai_df['price'].str.replace('[^0-9a-zA-Z., ]+', '', regex=True)
jaipur_df['price'] = chennai_df['price'].str.replace('[^0-9a-zA-Z., ]+', '', regex=True)
kolkata_df['price'] = chennai_df['price'].str.replace('[^0-9a-zA-Z., ]+', '', regex=True)


# List of columns to fill missing values with 'Unknown'
columns_to_fill = [
    'Registration Year', 'Power Steering', 'Anti Lock Braking System', 'bt', 'km',
    'ownerNo', 'oem', 'model', 'centralVariantId', 'variantName', 'Insurance Validity',
    'Fuel Type', 'Seats', 'Engine Displacement', 'Transmission', 'Year of Manufacture',
    'Max Power', 'Mileage', 'price'
]

# Fill missing values in each specified column
chennai_df[columns_to_fill] = chennai_df[columns_to_fill].fillna('Unknown')
bangalore_df[columns_to_fill] = bangalore_df[columns_to_fill].fillna('Unknown')
delhi_df[columns_to_fill] = delhi_df[columns_to_fill].fillna('Unknown')
hyderabad_df[columns_to_fill] = hyderabad_df[columns_to_fill].fillna('Unknown')
jaipur_df[columns_to_fill] = jaipur_df[columns_to_fill].fillna('Unknown')
kolkata_df[columns_to_fill] = kolkata_df[columns_to_fill].fillna('Unknown')


#removing & adding lakh & crore in new column
chennai_df['amt_type'] = chennai_df['price'].str.split().str[1]
chennai_df['price'] = chennai_df['price'].str.replace("Lakh", "").str.replace("Crore","")

bangalore_df['amt_type'] = bangalore_df['price'].str.split().str[1]
bangalore_df['price'] = bangalore_df['price'].str.replace("Lakh", "").str.replace("Crore","")

delhi_df['amt_type'] = delhi_df['price'].str.split().str[1]
delhi_df['price'] = delhi_df['price'].str.replace("Lakh", "").str.replace("Crore","")

hyderabad_df['amt_type'] = hyderabad_df['price'].str.split().str[1]
hyderabad_df['price'] = hyderabad_df['price'].str.replace("Lakh", "").str.replace("Crore","")

jaipur_df['amt_type'] = jaipur_df['price'].str.split().str[1]
jaipur_df['price'] = jaipur_df['price'].str.replace("Lakh", "").str.replace("Crore","")

kolkata_df['amt_type'] = kolkata_df['price'].str.split().str[1]
kolkata_df['price'] = kolkata_df['price'].str.replace("Lakh", "").str.replace("Crore","")


# Encoding Categorical Variables
# One-Hot Encoding for nominal categorical variables
chennai_df = pd.get_dummies(chennai_df, columns=['Fuel Type', 'Transmission'])
bangalore_df = pd.get_dummies(bangalore_df, columns=['Fuel Type', 'Transmission'])
delhi_df = pd.get_dummies(delhi_df, columns=['Fuel Type', 'Transmission'])
hyderabad_df = pd.get_dummies(hyderabad_df, columns=['Fuel Type', 'Transmission'])
jaipur_df = pd.get_dummies(jaipur_df, columns=['Fuel Type', 'Transmission'])
kolkata_df = pd.get_dummies(kolkata_df, columns=['Fuel Type', 'Transmission'])


# Example: If 'ownerNo' is ordinal
label_encoder = LabelEncoder()
chennai_df['ownerNo'] = label_encoder.fit_transform(chennai_df['ownerNo'])
bangalore_df['ownerNo'] = label_encoder.fit_transform(bangalore_df['ownerNo'])
delhi_df['ownerNo'] = label_encoder.fit_transform(delhi_df['ownerNo'])
hyderabad_df['ownerNo'] = label_encoder.fit_transform(hyderabad_df['ownerNo'])
jaipur_df['ownerNo'] = label_encoder.fit_transform(jaipur_df['ownerNo'])
kolkata_df['ownerNo'] = label_encoder.fit_transform(kolkata_df['ownerNo'])


# Mean imputation
# chennai_df1 = chennai_df.copy()
chennai_df['price'] = chennai_df['price'].str.replace(",", "")
chennai_df['price'] = chennai_df['price'].astype(float)
chennai_df['price'] = chennai_df['price'].fillna(chennai_df['price'].mean())

bangalore_df['price'] = bangalore_df['price'].str.strip()
bangalore_df['price'] = bangalore_df['price'].fillna(0)
bangalore_df['price'] = bangalore_df['price'].str.replace(",", "")
bangalore_df['price'] = bangalore_df['price'].str.replace("Unknown",'0')
bangalore_df['price'] = bangalore_df['price'].astype(float)
bangalore_df['price'] = bangalore_df['price'].fillna(bangalore_df['price'].mean())


delhi_df['price'] = delhi_df['price'].str.strip()
delhi_df['price'] = delhi_df['price'].fillna(0)
delhi_df['price'] = delhi_df['price'].str.replace(",", "")
delhi_df['price'] = delhi_df['price'].str.replace("Unknown",'0')
delhi_df['price'] = delhi_df['price'].astype(float)
delhi_df['price'] = delhi_df['price'].fillna(delhi_df['price'].mean())

hyderabad_df['price'] = hyderabad_df['price'].str.strip()
hyderabad_df['price'] = hyderabad_df['price'].fillna(0)
hyderabad_df['price'] = hyderabad_df['price'].str.replace(",", "")
hyderabad_df['price'] = hyderabad_df['price'].str.replace("Unknown",'0')
hyderabad_df['price'] = hyderabad_df['price'].astype(float)
hyderabad_df['price'] = hyderabad_df['price'].fillna(hyderabad_df['price'].mean())

jaipur_df['price'] = jaipur_df['price'].str.strip()
jaipur_df['price'] = jaipur_df['price'].fillna(0)
jaipur_df['price'] = jaipur_df['price'].str.replace(",", "")
jaipur_df['price'] = jaipur_df['price'].str.replace("Unknown",'0')
jaipur_df['price'] = jaipur_df['price'].astype(float)
jaipur_df['price'] = jaipur_df['price'].fillna(jaipur_df['price'].mean())

kolkata_df['price'] = kolkata_df['price'].str.strip()
kolkata_df['price'] = kolkata_df['price'].fillna(0)
kolkata_df['price'] = kolkata_df['price'].str.replace(",", "")
kolkata_df['price'] = kolkata_df['price'].str.replace("Unknown",'0')
kolkata_df['price'] = kolkata_df['price'].astype(float)
kolkata_df['price'] = kolkata_df['price'].fillna(kolkata_df['price'].mean())


# Save cleaned DataFrames to CSV
chennai_output_path = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/Save files/EDA data/Chennai_EDA_file.csv'
chennai_df.to_csv(chennai_output_path, index=False)

bangalore_output_path = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/Save files/EDA data/bangalore_EDA_file.csv'
bangalore_df.to_csv(bangalore_output_path, index=False)

delhi_output_path = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/Save files/EDA data/delhi_EDA_file.csv'
delhi_df.to_csv(delhi_output_path, index=False)

hyderabad_output_path = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/Save files/EDA data/hyderabad_EDA_file.csv'
hyderabad_df.to_csv(hyderabad_output_path, index=False)

jaipur_output_path = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/Save files/EDA data/jaipur_EDA_file.csv'
jaipur_df.to_csv(jaipur_output_path, index=False)

kolkata_output_path = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/Save files/EDA data/kolkata_EDA_file.csv'
kolkata_df.to_csv(kolkata_output_path, index=False)



#Train-Test Split:
selected_columns = [
    'price',
    'km',
    'ownerNo',
    'centralVariantId',
    'Seats',
    'Engine Displacement',
    'Year of Manufacture'
]

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat([chennai_df, bangalore_df, delhi_df, hyderabad_df, jaipur_df, kolkata_df], ignore_index=True)
combined_df = combined_df[['price','km','ownerNo','centralVariantId','Seats','Engine Displacement','Year of Manufacture']]


# Replace 'Unknown' with an empty string in specific columns
columns_to_clean = [
    'price', 'km', 'ownerNo', 'centralVariantId', 
    'Seats', 'Engine Displacement', 'Year of Manufacture'
]

for col in columns_to_clean:
    combined_df[col] = combined_df[col].replace("Unknown", "0")

combined_df['km'] = combined_df['km'].astype(str)
combined_df['km'] = combined_df['km'].str.replace(",", "")

# Convert the cleaned columns to numeric (float)
for col in columns_to_clean:
    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')


# Define features (X) and target (y)
X = combined_df.drop(columns=['price'])  # Drop the target column 'price'
y = combined_df['price']  # Target variable is 'price'

# Perform the train-test split with a 80-20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Model Selection and Training
# Initialize the RandomForestRegressor
model = RandomForestRegressor(random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model using RMSE (Root Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")


# Model Evaluation
# Performance Metrics

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Calculate R-squared (R²) Score
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²) Score: {r2:.2f}")
