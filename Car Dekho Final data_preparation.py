# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 22:35:08 2024

@author: Sai.Vigneshwar
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
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from scipy.stats import zscore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV

odf = pd.read_csv("C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/combined_data - Copy.csv")

cl = odf.columns

odf.drop(['Unnamed: 0'], axis=1, inplace=True)

ndf = odf.isna().sum().reset_index()
ndf.rename(columns={0:"values"}, inplace = True)

col_flt = ndf[ndf['values']>29000]

lst_col_flt = list(col_flt['index'])

count=0
for i in lst_col_flt:
    odf.drop(i, axis=1, inplace=True)
    count+=1
    # print(count)
    
odf['priceActual'] = odf['priceActual'].str.replace('[^0-9a-zA-Z., ]+', '', regex=True).str.strip()


def convert_to_real_number(s):
    s = s.replace(',', '')  # Remove commas
    if 'Lakh' in s:
        return float(s.replace('Lakh', '')) * 100000
    elif 'lakh' in s:
        return float(s.replace('Lakh', '')) * 100000
    elif 'Crore' in s:
        return float(s.replace('Crore', '')) * 10000000
    elif 'crore' in s:
        return float(s.replace('Crore', '')) * 10000000
    else:
        return float(s)
    
odf['priceActual'] = odf['priceActual'].fillna(0).astype(str)
odf['priceActual'] = odf['priceActual'].apply(convert_to_real_number)

odf['km'] = odf['km'].str.replace(',','').astype(int)

odf.loc[odf['amt_type'] == 'Lakh', 'price'] *= 100000
odf.loc[odf['amt_type'] == 'Crore', 'price'] *= 10000000

odf['Kms Driven']= odf['Kms Driven'].fillna(0).astype(str)
odf['Kms Driven'] = odf['Kms Driven'].str.replace("['\"]", "", regex=True).str.replace(',', '').str.strip().astype(int)

odf['Mileage'] = odf['Mileage'].replace("kmpl","").str.strip() 
odf['Mileage'] = odf['Mileage'].str.replace("km/kg","").str.replace(" kmpl","").replace("['\"]", "", regex=True)
odf['Mileage'] = odf['Mileage'].astype(float)


odf.drop(["amt_type"], axis=1, inplace=True)
odf.drop(['Ownership','owner'], axis=1, inplace=True)
odf.drop('Fuel Type', axis=1, inplace=True)
odf.drop("Transmission", axis=1, inplace=True)
odf.drop("Seats.1", axis=1, inplace=True)


yes_no_columns = [
    "Power Steering", "Power Windows Front", "Air Conditioner", "Heater",
    "Adjustable Head Lights", "Manually Adjustable Exterior Rear View Mirror",
    "Centeral Locking", "Fog Lights Front", "Anti Lock Braking System",
    "Cd Player", "Radio", "Power Adjustable Exterior Rear View Mirror",
    "Brake Assist"
]


# Replace feature names with 1 and blanks with 0
for col in yes_no_columns:
    odf[col] = odf[col].apply(lambda x: 1 if isinstance(x, str) and x.strip() != "" else 0)
    

odf.drop(['trendingText','trendingText2','trendingText3' ], axis=1, inplace=True)
nc = list(odf.columns[odf.dtypes != object])
non_nc = list(odf.columns[odf.dtypes == object])

#EDA for Mean, median & Mode
for col in nc:
    plt.figure(figsize=(8, 4))
    sns.histplot(odf[col],  kde=True, bins=30, alpha=0.7, label=col)
    plt.axvline(odf[col].mean(), color='red', linestyle='-.', linewidth=1, label='Mean')
    plt.axvline(odf[col].median(), color='green', linestyle='dashed', linewidth=1, label='Median')
    plt.legend()
    plt.show()
    
    
odf['Year of Manufacture'] = odf['modelYear']

odf['Mileage'].fillna(odf['Mileage'].median(), inplace=True)
print(odf['Mileage'].isna().sum())

odf['Engine Displacement'].fillna(odf['Engine Displacement'].median(), inplace=True)
odf['Engine'].fillna(odf['Engine'].median(), inplace=True)


plt.figure(figsize=(8, 4))
sns.histplot(odf['Mileage'], kde=True, bins=30, label="Mileage")
plt.axvline(odf['Mileage'].median(), color='green', linestyle='--', label='Median (after filling)')
plt.legend()
plt.title('Updated Distribution of Mileage')
plt.show()

odf['Max Power'] = odf['Max Power'].fillna("unknown")
odf['Torque'] = odf['Torque'].fillna("unknown")
odf['Wheel Size'] = odf['Wheel Size'].fillna("unknown")

odf['Seats'].fillna(odf['Seats'].median(), inplace=True)

ndf = odf.isna().sum().reset_index()
ndf.rename(columns={0:"values"}, inplace = True)

na_lst_col = list(ndf['index'])

for i in na_lst_col:
    odf[i] = odf[i].fillna("unknown")
    


odf.columns
#Feature engineering

odf['vehicle_age'] = 2024 - odf['Year of Manufacture']

# Select only numeric columns
numeric_df = odf.select_dtypes(include=['number'])

# Compute the correlation matrix
corr_matrix = numeric_df.corr()

df=odf.copy()


#Handling outliers and z score

# Function to cap outliers using the IQR method
def remove_outliers_iqr(df, column):
    
    # IQR method.
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Cap the outliers with lower and upper bounds
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df


# IQR method.

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Cap the outliers with lower and upper bounds
df['price'] = np.where(df['price'] < lower_bound, lower_bound, df['price'])
df['price'] = np.where(df['price'] > upper_bound, upper_bound, df['price'])    


# Function to remove outliers using the Z-score method
def remove_outliers_zscore(df, column, threshold=3):
    
    #Z-score method.
    
    z_scores = np.abs(zscore(df[column]))
    return df[z_scores < threshold]

# Function to process all numeric columns in a DataFrame
def handle_outliers(df, method='iqr', threshold=3):

    # Select only numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Loop through all numeric columns
    for col in numeric_columns:
        if method == 'iqr':
            df = remove_outliers_iqr(df, col)
            # print(f"Outliers capped for column: {col} using IQR method.")
        elif method == 'zscore':
            df = remove_outliers_zscore(df, col, threshold)
            # print(f"Outliers removed for column: {col} using Z-score method.")
        else:
            raise ValueError("Invalid method. Use 'iqr' or 'zscore'.")
    return df



# For IQR-based capping
df_cleaned_iqr = handle_outliers(df.copy(), method='iqr')

# For Z-score-based removal
df_cleaned_zscore = handle_outliers(df.copy(), method='zscore', threshold=3)

label_encoders = {}
for col in non_nc:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le  # Store encoders for later use

df['price'] = df['price'].astype(int)

X = df.drop(columns=['price', 'priceActual'])  # Drop target and ID columns
y = df['price']

variables = X.columns
nm_cols1 = [c for c in X.columns if X[c].dtype.name != 'object']
ct_cols = [c for c in X.columns if X[c].dtype.name == 'object']



#StandardScaler
scaler = StandardScaler()
X[nm_cols1] = scaler.fit_transform(X[nm_cols1])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")


mean_price = y_test.mean()
median_price = y_test.median()


#Correlations
correlations = X_train.join(y_train).corr()
print(correlations['price'].sort_values(ascending=False))


feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot top 10 features
feature_importances.head(10).plot.bar(x='Feature', y='Importance', legend=False, title='Top Features')
plt.show()

y_train_capped = np.clip(y_train, a_min=None, a_max=np.percentile(y_train, 99))


param_distributions = {
    'n_estimators': [100, 200],  # Fewer options
    'max_depth': [10, 20],       # Avoid `None` (unlimited depth)
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1]
}

random_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                            param_grid=param_distributions,
                            cv=2,
                            n_jobs=-1,
                            scoring='neg_mean_squared_error')


random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
print("Best parameters:", random_search.best_params_)

y_test_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2 = r2_score(y_test, y_test_pred)

print(f"New RMSE: {rmse}, New R²: {r2}")



# Bin continuous values into categories
bins = [0, 500000, 1000000, np.inf]
labels = ['Low', 'Medium', 'High']

y_test_binned = pd.cut(y_test, bins=bins, labels=labels)
y_pred_binned = pd.cut(y_test_pred, bins=bins, labels=labels)

# Generate confusion matrix
cm = confusion_matrix(y_test_binned, y_pred_binned, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Binned Predictions")
plt.show()

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Support Vector Regressor": SVR(),
    "K-Nearest Neighbors": KNeighborsRegressor()
}

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
    
    
dump(best_model, 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/final data/car_dheko_rf_best__model.joblib')




#Joblib model (Script for training streamlit)
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
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from scipy.stats import zscore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV

odf = pd.read_csv("C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/combined_data - Copy.csv")

model = load("C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/final data/car_dheko_rf_best__model.joblib")

cl = odf.columns

odf.drop(['Unnamed: 0'], axis=1, inplace=True)

ndf = odf.isna().sum().reset_index()
ndf.rename(columns={0:"values"}, inplace = True)

col_flt = ndf[ndf['values']>29000]


odf['priceActual'] = odf['priceActual'].str.replace('[^0-9a-zA-Z., ]+', '', regex=True).str.strip()

def convert_to_real_number(s):
    s = s.replace(',', '')  # Remove commas
    if 'Lakh' in s:
        return float(s.replace('Lakh', '')) * 100000
    elif 'lakh' in s:
        return float(s.replace('Lakh', '')) * 100000
    elif 'Crore' in s:
        return float(s.replace('Crore', '')) * 10000000
    elif 'crore' in s:
        return float(s.replace('Crore', '')) * 10000000
    else:
        return float(s)
    
odf['priceActual'] = odf['priceActual'].fillna(0).astype(str)
odf['priceActual'] = odf['priceActual'].apply(convert_to_real_number)

odf['km'] = odf['km'].str.replace(',','').astype(int)

odf.loc[odf['amt_type'] == 'Lakh', 'price'] *= 100000
odf.loc[odf['amt_type'] == 'Crore', 'price'] *= 10000000

odf['Kms Driven']= odf['Kms Driven'].fillna(0).astype(str)
odf['Kms Driven'] = odf['Kms Driven'].str.replace("['\"]", "", regex=True).str.replace(',', '').str.strip().astype(int)

odf['Mileage'] = odf['Mileage'].replace("kmpl","").str.strip() 
odf['Mileage'] = odf['Mileage'].str.replace("km/kg","").str.replace(" kmpl","").replace("['\"]", "", regex=True)
odf['Mileage'] = odf['Mileage'].astype(float)

lst_col_flt = list(col_flt['index'])

count=0
for i in lst_col_flt:
    odf.drop(i, axis=1, inplace=True)
    count+=1

odf['priceActual'] = odf['priceActual'].str.replace('[^0-9a-zA-Z., ]+', '', regex=True).str.strip()

def convert_to_real_number(s):
    s = s.replace(',', '')  # Remove commas
    if 'Lakh' in s:
        return float(s.replace('Lakh', '')) * 100000
    elif 'lakh' in s:
        return float(s.replace('Lakh', '')) * 100000
    elif 'Crore' in s:
        return float(s.replace('Crore', '')) * 10000000
    elif 'crore' in s:
        return float(s.replace('Crore', '')) * 10000000
    else:
        return float(s)
    
odf['priceActual'] = odf['priceActual'].fillna(0).astype(str)
odf['priceActual'] = odf['priceActual'].apply(convert_to_real_number)

odf['km'] = odf['km'].str.replace(',','').astype(int)

odf.loc[odf['amt_type'] == 'Lakh', 'price'] *= 100000
odf.loc[odf['amt_type'] == 'Crore', 'price'] *= 10000000

odf['Kms Driven']= odf['Kms Driven'].fillna(0).astype(str)
odf['Kms Driven'] = odf['Kms Driven'].str.replace("['\"]", "", regex=True).str.replace(',', '').str.strip().astype(int)

odf['Mileage'] = odf['Mileage'].replace("kmpl","").str.strip() 
odf['Mileage'] = odf['Mileage'].str.replace("km/kg","").str.replace(" kmpl","").replace("['\"]", "", regex=True)
odf['Mileage'] = odf['Mileage'].astype(float)


odf.drop(["amt_type"], axis=1, inplace=True)
odf.drop(['Ownership','owner'], axis=1, inplace=True)
odf.drop('Fuel Type', axis=1, inplace=True)
odf.drop("Transmission", axis=1, inplace=True)
odf.drop("Seats.1", axis=1, inplace=True)

yes_no_columns = [
    "Power Steering", "Power Windows Front", "Air Conditioner", "Heater",
    "Adjustable Head Lights", "Manually Adjustable Exterior Rear View Mirror",
    "Centeral Locking", "Fog Lights Front", "Anti Lock Braking System",
    "Cd Player", "Radio", "Power Adjustable Exterior Rear View Mirror",
    "Brake Assist"
]

# Replace feature names with 1 and blanks with 0
for col in yes_no_columns:
    odf[col] = odf[col].apply(lambda x: 1 if isinstance(x, str) and x.strip() != "" else 0)
    
    
odf.drop(['trendingText','trendingText2','trendingText3' ], axis=1, inplace=True)
nc = list(odf.columns[odf.dtypes != object])
non_nc = list(odf.columns[odf.dtypes == object])
    

odf['Year of Manufacture'] = odf['modelYear']

odf['Mileage'].fillna(odf['Mileage'].median(), inplace=True)

odf['Engine Displacement'].fillna(odf['Engine Displacement'].median(), inplace=True)
odf['Engine'].fillna(odf['Engine'].median(), inplace=True)
    
odf['Max Power'] = odf['Max Power'].fillna("unknown")
odf['Torque'] = odf['Torque'].fillna("unknown")
odf['Wheel Size'] = odf['Wheel Size'].fillna("unknown")

odf['Seats'].fillna(odf['Seats'].median(), inplace=True)

ndf = odf.isna().sum().reset_index()
ndf.rename(columns={0:"values"}, inplace = True)

na_lst_col = list(ndf['index'])

for i in na_lst_col:
    odf[i] = odf[i].fillna("unknown")
    
odf['vehicle_age'] = 2024 - odf['Year of Manufacture']

df = odf.copy()


label_encoders = {}
for col in non_nc:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le  # Store encoders for later use

df['price'] = df['price'].astype(int)

df.drop(['price','priceActual'], axis=1, inplace=True)

y_pred = model.predict(df)

odf['Predicted_Price'] = y_pred

odf['Predicted_Price'] = odf['Predicted_Price'].astype(str)
