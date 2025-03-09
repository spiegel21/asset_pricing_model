import pandas as pd
import numpy as np
import os

def preprocess():
    """Preprocess the raw stock data and save the processed data to a CSV file"""
    # Load the Excel file
    file_path = "data/stocks_prices.xlsx"
    excel_file = pd.ExcelFile(file_path)
    df_market = pd.read_csv('data/markets_return.csv')

    # Initialize an empty list to store dataframes
    dfs = []

    # Process each sheet
    for sheet_name in excel_file.sheet_names:
        # Read the main price data
        df_prices = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Drop the empty columns
        df_prices = df_prices.dropna(axis=1, how='all')

        # Extract the fillings data
        df_fillings = df_prices.iloc[:, 3:].dropna(how='all')
        df_prices = df_prices.iloc[:, :3]

        # Add the stock name from the sheet name
        df_prices['TICKER'] = sheet_name

        # Upper case all column names
        df_prices.columns = map(str.upper, df_prices.columns)
        df_fillings.columns = map(str.upper, df_fillings.columns)

        # Rename the columns
        df_fillings = df_fillings.rename(columns={'DATE.1': 'DATE'})

        # Make sure DATE is in datetime format
        df_prices['DATE'] = pd.to_datetime(df_prices['DATE'], format='%m%d%y')
        df_fillings['DATE'] = pd.to_datetime(df_fillings['DATE'], format='%m%d%y')
        df_market["DATE"] = pd.to_datetime(df_market["DATE"], format='%m/%d/%y')

        # Merge the fillings data
        df_full = df_prices.merge(df_fillings, on='DATE', how='outer')
        df_full[df_fillings.columns] = df_full[df_fillings.columns].sort_values('DATE').ffill()
        df_full = df_full.dropna(subset=df_prices.columns)

        # Merge the market data
        df_full = df_full.merge(df_market, on=['DATE', 'TICKER'], how='left')
        df_full[df_market.columns] = df_full[df_market.columns].sort_values('DATE').ffill()
        df_full = df_full.dropna(how='any')
        
        # Append to our list of dataframes
        dfs.append(df_full)

    # Concatenate all dataframes
    df_combined = pd.concat(dfs, ignore_index=True)

    # Save the processed dataframe
    df_combined.to_csv('data/processed_stock_data.csv', index=False)

    return None