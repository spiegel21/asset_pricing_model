import pandas as pd
import numpy as np
import statsmodels.api as sm

def fit_capm():
    """Fit the CAPM model to the stock data"""

    # Load data and rename variables
    df_returns = pd.read_csv('data/processed_stock_data.csv')

    # Remove the last year of data for evaluation
    df_eval = df_returns[df_returns['DATE'] >= '2019-01-01']
    df_returns = df_returns[df_returns['DATE'] < '2019-01-01']

    # Initialize CAPM betas dataframe and calculate betas
    betas = df_returns[["TICKER"]].drop_duplicates().reset_index(drop=True)
    betas['beta'] = np.nan
    betas['alphaM'] = np.nan
    betas['betaM'] = np.nan

    # Loop through stocks to estimate CAPM beta for each
    for i in betas['TICKER']:
        
        # Extract sub-dataframe
        subset = df_returns[df_returns['TICKER'] == i].dropna()
        
        # Prepare for regression
        x0 = np.array(subset['vwretd'])
        x1 = sm.add_constant(np.array(subset['vwretd']))
        y = np.array(subset['RET'])
        
        # Estimate CAPM
        capm = sm.OLS(y, x0).fit()
        
        # Estimate Market Model
        mktModel = sm.OLS(y, x1).fit()
        
        # Store parameters
        betas.loc[betas['TICKER'] == i, 'beta'] = capm.params[0]
        betas.loc[betas['TICKER'] == i, 'alphaM'] = mktModel.params[0]
        betas.loc[betas['TICKER'] == i, 'betaM'] = mktModel.params[1]

    # Evaluate the model
    df_eval = df_eval.merge(betas, on='TICKER', how='left')
    df_eval['expected_return_capm'] = df_eval['beta'] * df_eval['vwretd']
    df_eval['expected_return_mkt'] = df_eval['alphaM'] + df_eval['betaM'] * df_eval['vwretd']

    # Calculate the evaluation metrics
    df_eval['capm_error'] = df_eval['RET'] - df_eval['expected_return_capm']
    df_eval['mkt_error'] = df_eval['RET'] - df_eval['expected_return_mkt']

    # Rename the return column
    df_eval = df_eval.rename(columns={'RET': 'actual_return'})

    # Save the evaluation metrics
    df_eval[['DATE', 'TICKER', 'actual_return',
            'expected_return_capm', 'capm_error',
            'expected_return_mkt', 'mkt_error']].to_csv('output/capm_evaluation.csv', index=False)

    return None