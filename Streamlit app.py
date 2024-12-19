import pandas as pd
import numpy as np
import streamlit as st
from joblib import load
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# Load the trained model
@st.cache_data
def load_model():
    model_path = "C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/final data/car_dheko_rf_best__model.joblib"
    model = load(model_path)
    return model

# Function to clean the uploaded dataset
def preprocess_data(odf):
    global vf
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
    # odf['Kms Driven'] = odf['Kms Driven'].apply(lambda x:x.replace("'","")).replace(',','').str.strip().astype(int)
    odf['Kms Driven'] = odf['Kms Driven'].str.replace("['\"]", "", regex=True).str.replace(',', '').str.strip().astype(int)

    odf['Mileage'] = odf['Mileage'].replace("kmpl","").str.strip() 
    odf['Mileage'] = odf['Mileage'].str.replace("km/kg","").str.replace(" kmpl","").replace("['\"]", "", regex=True)
    odf['Mileage'] = odf['Mileage'].astype(float)

    # odf['Max Power'] = odf['Max Power'].replace("bhp","").str.strip()
    # odf['Torque'] = odf['Torque'].replace("Nm","").str.strip()

    odf.drop(["amt_type"], axis=1, inplace=True)
    odf.drop(['Ownership','owner'], axis=1, inplace=True)
    odf.drop('Fuel Type', axis=1, inplace=True)
    odf.drop("Transmission", axis=1, inplace=True)
    odf.drop("Seats.1", axis=1, inplace=True)

    # Yes/No Columns
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
    vf = odf.copy()
    label_encoders = {}
    for col in non_nc:
        le = LabelEncoder()
        odf[col] = le.fit_transform(odf[col].astype(str))
        label_encoders[col] = le  # Store encoders for later use

    return odf

# Streamlit UI
def main():
    st.title("Car Price Prediction Application 🚗")
    st.write("Upload your dataset for price prediction and download the results.")

    # File Upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    model = load_model()

    if uploaded_file:
        # Load data
        odf = pd.read_csv(uploaded_file)
        st.write("Data uploaded successfully!")
        st.write("Preview of uploaded data:")
        st.dataframe(odf.head())

        # Preprocess data
        st.write("Preprocessing data...")
        processed_df = preprocess_data(odf.copy())
        st.write("Data preprocessing completed.")

        # Drop unnecessary columns for prediction
        if 'price' in processed_df.columns:
            processed_df.drop(['price'], axis=1, inplace=True)
        if 'priceActual' in processed_df.columns:
            processed_df.drop(['priceActual'], axis=1, inplace=True)

        # Predict
        st.write("Running predictions...")
        y_pred = model.predict(processed_df)
        odf['Predicted_Price'] = y_pred
        odf['Predicted_Price'] = odf['Predicted_Price'].astype(str)

        st.write("Predictions completed. Preview of results:")
        st.dataframe(odf[['model','Predicted_Price']])
        
        # Filters Section
        st.sidebar.header("Filter Options")
        year_range = st.sidebar.slider("Select Manufacturing Year Range", 
                                       int(vf['modelYear'].min()), int(vf['modelYear'].max()), 
                                       (int(vf['modelYear'].min()), int(vf['modelYear'].max())))
        seats_filter = st.sidebar.multiselect("Select Number of Seats", sorted(vf['Seats'].unique()))

        filtered_df = vf[(vf['modelYear'] >= year_range[0]) & 
                          (vf['modelYear'] <= year_range[1])]
        
        if seats_filter:
            filtered_df = filtered_df[filtered_df['Seats'].isin(seats_filter)]

        st.write(f"Filtered Dataset: {len(filtered_df)} rows")
        st.dataframe(filtered_df.head())

        # Business Visualizations
        st.write("## Business Visualizations 📊")

        # 1. City-wise Count of Car Models
        st.subheader("City-wise Count of Car Models")
        city_count = filtered_df['city'].value_counts().reset_index()
        city_count.columns = ['City', 'Count']
        fig1 = px.bar(city_count, x='City', y='Count', title="City-wise Count of Car Models")
        st.plotly_chart(fig1)

        # 2. Mileage by Seats
        st.subheader("Average Mileage by Seats")
        mileage_by_seats = filtered_df.groupby('Seats')['Mileage'].mean().reset_index()
        fig2 = px.bar(mileage_by_seats, x='Seats', y='Mileage', title="Average Mileage by Number of Seats")
        st.plotly_chart(fig2)

        # 3. Year of Manufacture vs KM Driven
        st.subheader("Year of Manufacture vs KM Driven")
        fig3 = px.scatter(filtered_df, x='modelYear', y='Kms Driven', title="Year of Manufacture vs KM Driven")
        st.plotly_chart(fig3)

        # 4. Predicted Price Distribution
        st.subheader("Distribution of Predicted Prices")
        fig4 = px.histogram(odf, x="Predicted_Price", title="Distribution of Predicted Car Prices", nbins=30)
        st.plotly_chart(fig4)
        
        # Download option
        csv = odf.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predicted Results as CSV",
            data=csv,
            file_name="predicted_car_prices.csv",
            mime="text/csv",
        )
        st.success("Prediction and download completed successfully!")

if __name__ == "__main__":
    main()

