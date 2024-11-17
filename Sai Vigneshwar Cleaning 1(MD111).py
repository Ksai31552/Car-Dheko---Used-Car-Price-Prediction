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

# Load the dataset
file_path = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/chennai_cars.xlsx'
file_path2 = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/bangalore_cars.xlsx'
file_path3 = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/delhi_cars.xlsx'
file_path4 = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/hyderabad_cars.xlsx'
file_path5 = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/jaipur_cars.xlsx'
file_path6 = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/kolkata_cars.xlsx'

chennai_cars_df = pd.read_excel('C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/chennai_cars.xlsx')
bangalore_cars_df = pd.read_excel('C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/bangalore_cars.xlsx')
delhi_cars_df = pd.read_excel('C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/delhi_cars.xlsx')
hyderabad_cars_df = pd.read_excel('C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/hyderabad_cars.xlsx')
jaipur_cars_df = pd.read_excel('C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/jaipur_cars.xlsx')
kolkata_cars_df = pd.read_excel('C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/kolkata_cars.xlsx')


# spliting the data Seperately for every excel (Chennai 1 to 4)
chennai_cars_df.columns
dfl = pd.DataFrame()

#column 1

# Process each entry in 'new_car_detail'
for i in chennai_cars_df['new_car_detail']:
    # Convert the string representation of the dictionary to an actual dictionary
    result_dict = ast.literal_eval(i)
    
    # Handle the 'trendingText' field if it exists
    if 'trendingText' in result_dict:
        trending_dict = result_dict['trendingText']
        
        # Extract the desired values from the dictionary
        df = pd.DataFrame([{
            'trendingText': trending_dict.get('imgUrl', None),
            'trendingText2': trending_dict.get('heading', None),
            'trendingText3': trending_dict.get('desc', None),
            **result_dict  # Include the rest of the data
        }])
    else:
        # If 'trendingText' doesn't exist, create the dataframe with the rest of the data
        df = pd.DataFrame([result_dict])
    
    # Concatenate the processed DataFrame
    dfl = pd.concat([df, dfl], ignore_index=True)

# Add 'id' and 'city' columns
dfl['id'] = range(1, len(dfl) + 1)
dfl['city'] = 'Chennai'


#column 2
dfl2 = pd.DataFrame()
for i in chennai_cars_df['new_car_overview']:
    i = i.replace("'",'"')
    i = i.replace("None","null")
    data = json.loads(i)
    flat_data = {item['key']: item['value'] for item in data['top']}
    flat_data['heading'] = data['heading']
    
    # print(i)
    # jstr = chennai_cars_df['new_car_detail'][i]
    # result_dict = ast.literal_eval(i)
    df1 = pd.DataFrame([flat_data])
    dfl2 = pd.concat([df1,dfl2])
 
# Replace 'Seats' with an empty string in the 'Seat' column
if 'Seats' in dfl2.columns:
    dfl2['Seats'] = dfl2['Seats'].str.replace('Seats', '', regex=False)
    
# Replace 'kms' with an empty string in the 'kms driven' column
if 'Kms Driven' in dfl2.columns:
    dfl2['Kms Driven'] = dfl2['Kms Driven'].str.replace('Kms', '', regex=False)    

# Replace 'CC' with an empty string in the 'Engine Displacement' column
if 'Engine Displacement' in dfl2.columns:
    dfl2['Engine Displacement'] = dfl2['Engine Displacement'].str.replace('cc', '', regex=False)     

#dfl['id'] = range(1, len(dfl) + 1)
#dfl['city'] = 'Chennai'



#column 3
dfl3 = []
for i in chennai_cars_df['new_car_feature']:
    i = i.replace("'", '"')
    i = i.replace("None", "null")
    data = json.loads(i)
    flat_data1 = {item['value']: item['value'] for item in data['top']}
    flat_data1['heading'] = data['heading']
    
    # Create a DataFrame for the current entry
    df2 = pd.DataFrame([flat_data1])
    
    # Append the DataFrame to the list
    dfl3.append(df2)

# Concatenate all DataFrames at once
dfl3 = pd.concat(dfl3, ignore_index=True)



#column 4
dfl4 = []
for i in chennai_cars_df['new_car_specs']:
    i = i.replace("'", '"')
    i = i.replace("None", "null")
    data = json.loads(i)
    flat_data2 = {item['key']: item['value'] for item in data['top']}
    flat_data2['heading'] = data['heading']
    
    # Create a DataFrame for the current entry
    df3 = pd.DataFrame([flat_data2])
    
    # Append the DataFrame to the list
    dfl4.append(df3)

# Concatenate all DataFrames at once
dfl4 = pd.concat(dfl4, ignore_index=True)

# Replace 'CC' with an empty string in the 'Engine' column
if 'Engine' in dfl4.columns:
    dfl4['Engine'] = dfl4['Engine'].str.replace('CC', '', regex=False)


# Reset the index for each DataFrame to ensure unique indexing
dfl.reset_index(drop=True, inplace=True)
dfl2.reset_index(drop=True, inplace=True)
dfl3.reset_index(drop=True, inplace=True)
dfl4.reset_index(drop=True, inplace=True)

# Merge all the dataframes along the columns (axis=1)
chennai_cars_df = pd.concat([dfl, dfl2, dfl3, dfl4], axis=1)

# Save the final dataframe to an Excel file
output_file = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/Save files/chennai_cars.csv'
chennai_cars_df.to_csv(output_file, index=False)






# spliting the data Seperately for every excel (bangalore)

#column 1
bangalore_cars_df.columns
df1 = pd.DataFrame()

# Process each entry in 'new_car_detail'
for i in bangalore_cars_df['new_car_detail']:
    # Convert the string representation of the dictionary to an actual dictionary
    result_dict = ast.literal_eval(i)
    
    # Handle the 'trendingText' field if it exists
    if 'trendingText' in result_dict:
        trending_dict = result_dict['trendingText']
        
        # Extract the desired values from the dictionary
        df = pd.DataFrame([{
            'trendingText': trending_dict.get('imgUrl', None),
            'trendingText2': trending_dict.get('heading', None),
            'trendingText3': trending_dict.get('desc', None),
            **result_dict  # Include the rest of the data
        }])
    else:
        # If 'trendingText' doesn't exist, create the dataframe with the rest of the data
        df = pd.DataFrame([result_dict])
    
    # Concatenate the processed DataFrame
    dfl = pd.concat([df, dfl], ignore_index=True)

dfl['id'] = range(1, len(dfl) + 1)
dfl['city'] = 'Bangalore'

#column 2
dfl2 = pd.DataFrame()
for i in bangalore_cars_df['new_car_overview']:
    i = i.replace("'",'"')
    i = i.replace("None","null")
    data = json.loads(i)
    flat_data = {item['key']: item['value'] for item in data['top']}
    flat_data['heading'] = data['heading']
    
    # print(i)
    # jstr = chennai_cars_df['new_car_detail'][i]
    # result_dict = ast.literal_eval(i)
    df1 = pd.DataFrame([flat_data])
    dfl2 = pd.concat([df1,dfl2])
    
# Replace 'Seats' with an empty string in the 'Seat' column
if 'Seats' in dfl2.columns:
    dfl2['Seats'] = dfl2['Seats'].str.replace('Seats', '', regex=False)
    
# Replace 'kms' with an empty string in the 'kms driven' column
if 'Kms Driven' in dfl2.columns:
    dfl2['Kms Driven'] = dfl2['Kms Driven'].str.replace('Kms', '', regex=False)

# Replace 'CC' with an empty string in the 'Engine Displacement' column
if 'Engine Displacement' in dfl2.columns:
    dfl2['Engine Displacement'] = dfl2['Engine Displacement'].str.replace('cc', '', regex=False)  



#column 3

dfl3 = []
for i in bangalore_cars_df['new_car_feature']:
    i = i.replace("'", '"')
    i = i.replace("None", "null")
    data = json.loads(i)
    flat_data1 = {item['value']: item['value'] for item in data['top']}
    flat_data1['heading'] = data['heading']
    
    # Create a DataFrame for the current entry
    df2 = pd.DataFrame([flat_data1])
    
    # Append the DataFrame to the list
    dfl3.append(df2)

# Concatenate all DataFrames at once
dfl3 = pd.concat(dfl3, ignore_index=True)


#column 4
dfl4 = []
for i in bangalore_cars_df['new_car_specs']:
    i = i.replace("'", '"')
    i = i.replace("None", "null")
    data = json.loads(i)
    flat_data2 = {item['key']: item['value'] for item in data['top']}
    flat_data2['heading'] = data['heading']
    
    # Create a DataFrame for the current entry
    df3 = pd.DataFrame([flat_data2])
    
    # Append the DataFrame to the list
    dfl4.append(df3)

# Concatenate all DataFrames at once
dfl4 = pd.concat(dfl4, ignore_index=True)

# Replace 'CC' with an empty string in the 'Engine' column
if 'Engine' in dfl4.columns:
    dfl4['Engine'] = dfl4['Engine'].str.replace('CC', '', regex=False)


# Reset the index for each DataFrame to ensure unique indexing
dfl.reset_index(drop=True, inplace=True)
dfl2.reset_index(drop=True, inplace=True)
dfl3.reset_index(drop=True, inplace=True)
dfl4.reset_index(drop=True, inplace=True)

# Merge all the dataframes along the columns (axis=1)
bangalore_cars_df = pd.concat([dfl, dfl2, dfl3, dfl4], axis=1)

# Save the final dataframe to an Excel file
output_file = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/Save files/bangalore_cars.csv'
bangalore_cars_df.to_csv(output_file, index=False)




# spliting the data Seperately for every excel (Delhi)

#column 1
delhi_cars_df.columns
df1 = pd.DataFrame()

# Process each entry in 'new_car_detail'
for i in delhi_cars_df['new_car_detail']:
    # Convert the string representation of the dictionary to an actual dictionary
    result_dict = ast.literal_eval(i)
    
    # Handle the 'trendingText' field if it exists
    if 'trendingText' in result_dict:
        trending_dict = result_dict['trendingText']
        
        # Extract the desired values from the dictionary
        df = pd.DataFrame([{
            'trendingText': trending_dict.get('imgUrl', None),
            'trendingText2': trending_dict.get('heading', None),
            'trendingText3': trending_dict.get('desc', None),
            **result_dict  # Include the rest of the data
        }])
    else:
        # If 'trendingText' doesn't exist, create the dataframe with the rest of the data
        df = pd.DataFrame([result_dict])
    
    # Concatenate the processed DataFrame
    dfl = pd.concat([df, dfl], ignore_index=True)

dfl['id'] = range(1, len(dfl) + 1)
dfl['city'] = 'Delhi'


#column 2
dfl2 = pd.DataFrame()
for i in delhi_cars_df['new_car_overview']:
    i = i.replace("'",'"')
    i = i.replace("None","null")
    data = json.loads(i)
    flat_data = {item['key']: item['value'] for item in data['top']}
    flat_data['heading'] = data['heading']
    
    # print(i)
    # jstr = chennai_cars_df['new_car_detail'][i]
    # result_dict = ast.literal_eval(i)
    df1 = pd.DataFrame([flat_data])
    dfl2 = pd.concat([df1,dfl2])
   

# Replace 'Seats' with an empty string in the 'Seat' column
if 'Seats' in dfl2.columns:
    dfl2['Seats'] = dfl2['Seats'].str.replace('Seats', '', regex=False)
    
# Replace 'kms' with an empty string in the 'kms driven' column
if 'Kms Driven' in dfl2.columns:
    dfl2['Kms Driven'] = dfl2['Kms Driven'].str.replace('Kms', '', regex=False)    

# Replace 'CC' with an empty string in the 'Engine Displacement' column
if 'Engine Displacement' in dfl2.columns:
    dfl2['Engine Displacement'] = dfl2['Engine Displacement'].str.replace('cc', '', regex=False)
    
    

#column 3
dfl3 = []
for i in delhi_cars_df['new_car_feature']:
    i = i.replace("'", '"')
    i = i.replace("None", "null")
    data = json.loads(i)
    flat_data1 = {item['value']: item['value'] for item in data['top']}
    flat_data1['heading'] = data['heading']
    
    # Create a DataFrame for the current entry
    df2 = pd.DataFrame([flat_data1])
    
    # Append the DataFrame to the list
    dfl3.append(df2)

# Concatenate all DataFrames at once
dfl3 = pd.concat(dfl3, ignore_index=True)



#column 4
dfl4 = []
for i in delhi_cars_df['new_car_specs']:
    i = i.replace("'", '"')
    i = i.replace("None", "null")
    data = json.loads(i)
    flat_data2 = {item['key']: item['value'] for item in data['top']}
    flat_data2['heading'] = data['heading']
    
    # Create a DataFrame for the current entry
    df3 = pd.DataFrame([flat_data2])
    
    # Append the DataFrame to the list
    dfl4.append(df3)

# Concatenate all DataFrames at once
dfl4 = pd.concat(dfl4, ignore_index=True)

# Replace 'CC' with an empty string in the 'Engine' column
if 'Engine' in dfl4.columns:
    dfl4['Engine'] = dfl4['Engine'].str.replace('CC', '', regex=False)


# Reset the index for each DataFrame to ensure unique indexing
dfl.reset_index(drop=True, inplace=True)
dfl2.reset_index(drop=True, inplace=True)
dfl3.reset_index(drop=True, inplace=True)
dfl4.reset_index(drop=True, inplace=True)

# Merge all the dataframes along the columns (axis=1)
delhi_cars_df = pd.concat([dfl, dfl2, dfl3, dfl4], axis=1)

# Save the final dataframe to an Excel file
output_file = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/Save files/delhi_cars.csv'
delhi_cars_df.to_csv(output_file, index=False)




# spliting the data Seperately for every excel (Hyderabad)

#column 1
hyderabad_cars_df.columns
df1 = pd.DataFrame()

# Process each entry in 'new_car_detail'
for i in hyderabad_cars_df['new_car_detail']:
    # Convert the string representation of the dictionary to an actual dictionary
    result_dict = ast.literal_eval(i)
    
    # Handle the 'trendingText' field if it exists
    if 'trendingText' in result_dict:
        trending_dict = result_dict['trendingText']
        
        # Extract the desired values from the dictionary
        df = pd.DataFrame([{
            'trendingText': trending_dict.get('imgUrl', None),
            'trendingText2': trending_dict.get('heading', None),
            'trendingText3': trending_dict.get('desc', None),
            **result_dict  # Include the rest of the data
        }])
    else:
        # If 'trendingText' doesn't exist, create the dataframe with the rest of the data
        df = pd.DataFrame([result_dict])
    
    # Concatenate the processed DataFrame
    dfl = pd.concat([df, dfl], ignore_index=True)
  
dfl['id'] = range(1, len(dfl) + 1)
dfl['city'] = 'Hyderabad'    



#column 2
dfl2 = pd.DataFrame()
for i in hyderabad_cars_df['new_car_overview']:
    i = i.replace("'",'"')
    i = i.replace("None","null")
    data = json.loads(i)
    flat_data = {item['key']: item['value'] for item in data['top']}
    flat_data['heading'] = data['heading']
    
    # print(i)
    # jstr = chennai_cars_df['new_car_detail'][i]
    # result_dict = ast.literal_eval(i)
    df1 = pd.DataFrame([flat_data])
    dfl2 = pd.concat([df1,dfl2])
 
# Replace 'Seats' with an empty string in the 'Seat' column
if 'Seats' in dfl2.columns:
    dfl2['Seats'] = dfl2['Seats'].str.replace('Seats', '', regex=False)
    
# Replace 'kms' with an empty string in the 'kms driven' column
if 'Kms Driven' in dfl2.columns:
    dfl2['Kms Driven'] = dfl2['Kms Driven'].str.replace('Kms', '', regex=False)

# Replace 'CC' with an empty string in the 'Engine Displacement' column
if 'Engine Displacement' in dfl2.columns:
    dfl2['Engine Displacement'] = dfl2['Engine Displacement'].str.replace('cc', '', regex=False)



#column 3
dfl3 = []
for i in hyderabad_cars_df['new_car_feature']:
    i = i.replace("'", '"')
    i = i.replace("None", "null")
    data = json.loads(i)
    flat_data1 = {item['value']: item['value'] for item in data['top']}
    flat_data1['heading'] = data['heading']
    
    # Create a DataFrame for the current entry
    df2 = pd.DataFrame([flat_data1])
    
    # Append the DataFrame to the list
    dfl3.append(df2)

# Concatenate all DataFrames at once
dfl3 = pd.concat(dfl3, ignore_index=True)



#column 4
dfl4 = []
for i in hyderabad_cars_df['new_car_specs']:
    i = i.replace("'", '"')
    i = i.replace("None", "null")
    data = json.loads(i)
    flat_data2 = {item['key']: item['value'] for item in data['top']}
    flat_data2['heading'] = data['heading']
    
    # Create a DataFrame for the current entry
    df3 = pd.DataFrame([flat_data2])
    
    # Append the DataFrame to the list
    dfl4.append(df3)

# Concatenate all DataFrames at once
dfl4 = pd.concat(dfl4, ignore_index=True)

# Replace 'CC' with an empty string in the 'Engine' column
if 'Engine' in dfl4.columns:
    dfl4['Engine'] = dfl4['Engine'].str.replace('CC', '', regex=False)
    
    
# Reset the index for each DataFrame to ensure unique indexing
dfl.reset_index(drop=True, inplace=True)
dfl2.reset_index(drop=True, inplace=True)
dfl3.reset_index(drop=True, inplace=True)
dfl4.reset_index(drop=True, inplace=True)

# Merge all the dataframes along the columns (axis=1)
hyderabad_cars_df = pd.concat([dfl, dfl2, dfl3, dfl4], axis=1)

# Save the final dataframe to an Excel file
output_file = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/Save files/hyderabad_cars.csv'
hyderabad_cars_df.to_csv(output_file, index=False)




# spliting the data Seperately for every excel (Jaipur)

#column 1
jaipur_cars_df.columns
df1 = pd.DataFrame()

# Process each entry in 'new_car_detail'
for i in jaipur_cars_df['new_car_detail']:
    # Convert the string representation of the dictionary to an actual dictionary
    result_dict = ast.literal_eval(i)
    
    # Handle the 'trendingText' field if it exists
    if 'trendingText' in result_dict:
        trending_dict = result_dict['trendingText']
        
        # Extract the desired values from the dictionary
        df = pd.DataFrame([{
            'trendingText': trending_dict.get('imgUrl', None),
            'trendingText2': trending_dict.get('heading', None),
            'trendingText3': trending_dict.get('desc', None),
            **result_dict  # Include the rest of the data
        }])
    else:
        # If 'trendingText' doesn't exist, create the dataframe with the rest of the data
        df = pd.DataFrame([result_dict])
    
    # Concatenate the processed DataFrame
    dfl = pd.concat([df, dfl], ignore_index=True)
  
dfl['id'] = range(1, len(dfl) + 1)
dfl['city'] = 'Jaipur'    
  


#column 2
dfl2 = pd.DataFrame()
for i in jaipur_cars_df['new_car_overview']:
    i = i.replace("'",'"')
    i = i.replace("None","null")
    data = json.loads(i)
    flat_data = {item['key']: item['value'] for item in data['top']}
    flat_data['heading'] = data['heading']
    
    # print(i)
    # jstr = chennai_cars_df['new_car_detail'][i]
    # result_dict = ast.literal_eval(i)
    df1 = pd.DataFrame([flat_data])
    dfl2 = pd.concat([df1,dfl2])

# Replace 'Seats' with an empty string in the 'Seat' column
if 'Seats' in dfl2.columns:
    dfl2['Seats'] = dfl2['Seats'].str.replace('Seats', '', regex=False)
    
# Replace 'kms' with an empty string in the 'kms driven' column
if 'Kms Driven' in dfl2.columns:
    dfl2['Kms Driven'] = dfl2['Kms Driven'].str.replace('Kms', '', regex=False)
    
# Replace 'CC' with an empty string in the 'Engine Displacement' column
if 'Engine Displacement' in dfl2.columns:
    dfl2['Engine Displacement'] = dfl2['Engine Displacement'].str.replace('cc', '', regex=False)    


#column 3
dfl3 = []
for i in jaipur_cars_df['new_car_feature']:
    i = i.replace("'", '"')
    i = i.replace("None", "null")
    data = json.loads(i)
    flat_data1 = {item['value']: item['value'] for item in data['top']}
    flat_data1['heading'] = data['heading']
    
    # Create a DataFrame for the current entry
    df2 = pd.DataFrame([flat_data1])
    
    # Append the DataFrame to the list
    dfl3.append(df2)

# Concatenate all DataFrames at once
dfl3 = pd.concat(dfl3, ignore_index=True)




#column 4
dfl4 = []
for i in jaipur_cars_df['new_car_specs']:
    i = i.replace("'", '"')
    i = i.replace("None", "null")
    data = json.loads(i)
    flat_data2 = {item['key']: item['value'] for item in data['top']}
    flat_data2['heading'] = data['heading']
    
    # Create a DataFrame for the current entry
    df3 = pd.DataFrame([flat_data2])
    
    # Append the DataFrame to the list
    dfl4.append(df3)

# Concatenate all DataFrames at once
dfl4 = pd.concat(dfl4, ignore_index=True)

# Replace 'CC' with an empty string in the 'Engine' column
if 'Engine' in dfl4.columns:
    dfl4['Engine'] = dfl4['Engine'].str.replace('CC', '', regex=False)

# Reset the index for each DataFrame to ensure unique indexing
dfl.reset_index(drop=True, inplace=True)
dfl2.reset_index(drop=True, inplace=True)
dfl3.reset_index(drop=True, inplace=True)
dfl4.reset_index(drop=True, inplace=True)

# Merge all the dataframes along the columns (axis=1)
jaipur_cars_df = pd.concat([dfl, dfl2, dfl3, dfl4], axis=1)

# Save the final dataframe to an Excel file
output_file = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/Save files/jaipur_cars.csv'
jaipur_cars_df.to_csv(output_file, index=False)




# spliting the data Seperately for every excel (Kolkata)

#column 1
kolkata_cars_df.columns
df1 = pd.DataFrame()

# Process each entry in 'new_car_detail'
for i in kolkata_cars_df['new_car_detail']:
    # Convert the string representation of the dictionary to an actual dictionary
    result_dict = ast.literal_eval(i)
    
    # Handle the 'trendingText' field if it exists
    if 'trendingText' in result_dict:
        trending_dict = result_dict['trendingText']
        
        # Extract the desired values from the dictionary
        df = pd.DataFrame([{
            'trendingText': trending_dict.get('imgUrl', None),
            'trendingText2': trending_dict.get('heading', None),
            'trendingText3': trending_dict.get('desc', None),
            **result_dict  # Include the rest of the data
        }])
    else:
        # If 'trendingText' doesn't exist, create the dataframe with the rest of the data
        df = pd.DataFrame([result_dict])
    
    # Concatenate the processed DataFrame
    dfl = pd.concat([df, dfl], ignore_index=True)
    
dfl['id'] = range(1, len(dfl) + 1)
dfl['city'] = 'Kolkata'
    


#column 2
dfl2 = pd.DataFrame()
for i in kolkata_cars_df['new_car_overview']:
    i = i.replace("'",'"')
    i = i.replace("None","null")
    data = json.loads(i)
    flat_data = {item['key']: item['value'] for item in data['top']}
    flat_data['heading'] = data['heading']
    
    # print(i)
    # jstr = chennai_cars_df['new_car_detail'][i]
    # result_dict = ast.literal_eval(i)
    df1 = pd.DataFrame([flat_data])
    dfl2 = pd.concat([df1,dfl2])
    
# Replace 'Seats' with an empty string in the 'Seat' column
if 'Seats' in dfl2.columns:
    dfl2['Seats'] = dfl2['Seats'].str.replace('Seats', '', regex=False)
    
# Replace 'kms' with an empty string in the 'kms driven' column
if 'Kms Driven' in dfl2.columns:
    dfl2['Kms Driven'] = dfl2['Kms Driven'].str.replace('Kms', '', regex=False)

# Replace 'CC' with an empty string in the 'Engine Displacement' column
if 'Engine Displacement' in dfl2.columns:
    dfl2['Engine Displacement'] = dfl2['Engine Displacement'].str.replace('cc', '', regex=False)    


#column 3
dfl3 = []
for i in kolkata_cars_df['new_car_feature']:
    i = i.replace("'", '"')
    i = i.replace("None", "null")
    data = json.loads(i)
    flat_data1 = {item['value']: item['value'] for item in data['top']}
    flat_data1['heading'] = data['heading']
    
    # Create a DataFrame for the current entry
    df2 = pd.DataFrame([flat_data1])
    
    # Append the DataFrame to the list
    dfl3.append(df2)

# Concatenate all DataFrames at once
dfl3 = pd.concat(dfl3, ignore_index=True)




#column 4
dfl4 = []
for i in kolkata_cars_df['new_car_specs']:
    i = i.replace("'", '"')
    i = i.replace("None", "null")
    data = json.loads(i)
    flat_data2 = {item['key']: item['value'] for item in data['top']}
    flat_data2['heading'] = data['heading']
    
    # Create a DataFrame for the current entry
    df3 = pd.DataFrame([flat_data2])
    
    # Append the DataFrame to the list
    dfl4.append(df3)

# Concatenate all DataFrames at once
dfl4 = pd.concat(dfl4, ignore_index=True)


#dfl['Mileage'] = range(1, len(dfl) + 1)
#dfl['KMPL'] = 'kmpl'

# Replace 'KMPL' with an empty string in the 'Mileage' column
#if 'Mileage' in dfl4.columns:
#    dfl4['Mileage'] = dfl4['Mileage'].str.replace('kmpl', '', regex=False)


# Replace 'CC' with an empty string in the 'Engine' column
if 'Engine' in dfl4.columns:
    dfl4['Engine'] = dfl4['Engine'].str.replace('CC', '', regex=False)
    
# Optional: If you want to trim any extra spaces left after removal
#dfl4['Engine'] = dfl4['Engine'].str.strip()

# Reset the index for each DataFrame to ensure unique indexing
dfl.reset_index(drop=True, inplace=True)
dfl2.reset_index(drop=True, inplace=True)
dfl3.reset_index(drop=True, inplace=True)
dfl4.reset_index(drop=True, inplace=True)

# Merge all the dataframes along the columns (axis=1)
kolkata_cars_df = pd.concat([dfl, dfl2, dfl3, dfl4], axis=1)

# Save the final dataframe to an Excel file
output_file = 'C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Car Dekho/City wise Car data/Save files/kolkata_cars.csv'
kolkata_cars_df.to_csv(output_file, index=False)