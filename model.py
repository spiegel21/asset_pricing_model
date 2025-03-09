import pandas as pd
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns

def train_model():
    """Train the CatBoost model"""
    # Load data
    df_returns = pd.read_csv('data/processed_stock_data.csv')

    # Convert all numeric columns to float
    numeric_cols = list(set(df_returns.columns) - set(["DATE", "TICKER"]))
    df_returns[numeric_cols] = df_returns[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Get all price related columns
    price_cols = list(set(df_returns.columns) - set(["DATE", "TICKER", "vwretd", "RET"]))

    # Take the average of price cols for each ticker and month
    df_returns["MONTH_YEAR"] = df_returns["DATE"].str[:7]
    df_returns[price_cols] = df_returns.groupby(["TICKER", "MONTH_YEAR"])[price_cols].transform("mean")
    df_returns = df_returns.drop(columns=["MONTH_YEAR"])

    # Lag the price columns
    df_returns[price_cols] = df_returns.groupby('TICKER')[price_cols].shift(1)
    df_returns = df_returns.dropna()

    # Remove the last year of data for evaluation
    df_test = df_returns[df_returns['DATE'] >= '2019-01-01']
    df_val = df_returns[(df_returns['DATE'] >= '2018-01-01') & (df_returns['DATE'] < '2019-01-01')]
    df_train = df_returns[df_returns['DATE'] < '2018-01-01']

    # Features to use (all except DATE and RET)
    feature_cols = [col for col in df_train.columns if col not in ['DATE', 'RET']]

    # Prepare features and target
    X_train = df_train[feature_cols]
    y_train = df_train['RET']
    X_val = df_val[feature_cols]
    y_val = df_val['RET']

    # Train CatBoost model with early stopping
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.01,
        loss_function='RMSE',
        verbose=50,
        cat_features=['TICKER']
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

    # Make predictions on the test set
    X_eval = df_test[feature_cols]
    y_eval = df_test['RET']
    y_pred = model.predict(X_eval)

    # Create evaluation dataframe
    eval_results = pd.DataFrame({
        'DATE': df_test['DATE'],
        'TICKER': df_test['TICKER'],
        'actual_return': y_eval,
        'predicted_return': y_pred
    })

    # Calculate errors
    eval_results['catboost_error'] = eval_results['actual_return'] - eval_results['predicted_return']

    # Save the evaluation results
    eval_results.to_csv('output/catboost_evaluation.csv', index=False)

    feature_importance = model.get_feature_importance(prettified=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importances', y='Feature Id', data=feature_importance.head(10))
    plt.title('Top 10 Feature Importances')
    plt.show()

    return None
    