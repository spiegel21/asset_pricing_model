import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_models():
    """Evaluate the models and compare the results"""

    # Compare to simple CAPM model
    capm_eval = pd.read_csv('output/capm_evaluation.csv')
    catboost_eval = pd.read_csv('output/catboost_evaluation.csv')

    # Merge the results
    comparison = pd.merge(
        catboost_eval, 
        capm_eval, 
        on=['DATE', 'TICKER', 'actual_return'],
        how='inner'
    )

    # Compare errors
    catboost_mse = np.mean(comparison['catboost_error']**2)
    capm_mse = np.mean(comparison['capm_error']**2)
    mkt_mse = np.mean(comparison['mkt_error']**2)

    print(f'CatBoost MSE: {catboost_mse:.6f}')
    print(f'CAPM MSE: {capm_mse:.6f}')
    print(f'Market Model MSE: {mkt_mse:.6f}')

    # Determine which model is better based on MSE
    if catboost_mse < capm_mse and catboost_mse < mkt_mse:
        print('CatBoost model is better.')
    elif capm_mse < catboost_mse and capm_mse < mkt_mse:
        print('CAPM model is better.')
    else:
        print('Market model is better.')

    # Save the evaluation results
    comparison.to_csv('output/overall_evaluation.csv', index=False)

    return None