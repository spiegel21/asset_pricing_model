import pandas as pd
import numpy as np
import os

# Load the Excel file
file_path = "data/MGT189_DATA.xlsx"
excel_file = pd.ExcelFile(file_path)

# Initialize an empty list to store dataframes
dfs = []

# Process each sheet
for sheet_name in excel_file.sheet_names:
    # Read the main price data
    df_prices = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Add the stock name from the sheet name
    df_prices['Symbol'] = sheet_name
    
    # Drop the empty columns
    df_prices = df_prices.dropna(axis=1, how='all')

    # Extract the fillings data
    df_fillings = df_prices.iloc[:, 3:]
    df_prices = df_prices.iloc[:, :3]

    # Upper case all column names
    df_prices.columns = map(str.upper, df_prices.columns)
    df_fillings.columns = map(str.upper, df_fillings.columns)

    # Rename the columns
    df_fillings = df_fillings.rename(columns={'DATE.1': 'DATE'})

    # Make sure DATE is in datetime format
    df_prices['DATE'] = pd.to_datetime(df_prices['DATE'], format='%m%d%y')
    df_fillings['DATE'] = pd.to_datetime(df_fillings['DATE'], format='%m%d%y')

    # Merge the two dataframes
    df_full = df_prices.merge(df_fillings, on='DATE', how='outer')

    # Forward fill the missing values
    df_full = df_full.sort_values('DATE')
    df_full = df_full.fillna(method='ffill')
    
    # Append to our list of dataframes
    dfs.append(df_full)

# Concatenate all dataframes
df_combined = pd.concat(dfs, ignore_index=True)

# Save the processed dataframe
df_combined.to_csv('data/processed_stock_data.csv', index=False)