import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

def train_model():
    """Train the CatBoost model"""
    # Load data
    df_returns = pd.read_csv('data/processed_stock_data.csv')

    # Convert all numeric columns to float
    numeric_cols = list(set(df_returns.columns) - set(["DATE", "TICKER"]))
    df_returns[numeric_cols] = df_returns[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Lag all columns related to price in order to avoid lookahead bias
    price_cols = list(set(df_returns.columns) - set(["DATE", "TICKER", "vwretd"]))
    df_returns[price_cols] = df_returns.groupby('TICKER')[price_cols].shift(1)
    df_returns = df_returns.dropna()

    # Remove the last year of data for evaluation
    df_eval = df_returns[df_returns['DATE'] >= '2019-01-01']
    df_train = df_returns[df_returns['DATE'] < '2019-01-01']

    # Features to use (all except DATE and RET)
    feature_cols = [col for col in df_train.columns if col not in ['DATE', 'RET']]

    # Prepare features and target
    X = df_train[feature_cols]
    y = df_train['RET']

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train CatBoost model with early stopping
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        verbose=0,
        cat_features=['TICKER']
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

    # Make predictions on the evaluation set
    X_eval = df_eval[feature_cols]
    y_eval = df_eval['RET']
    y_pred = model.predict(X_eval)

    # Create evaluation dataframe
    eval_results = pd.DataFrame({
        'DATE': df_eval['DATE'],
        'TICKER': df_eval['TICKER'],
        'actual_return': y_eval,
        'predicted_return': y_pred
    })

    # Calculate errors
    eval_results['catboost_error'] = eval_results['actual_return'] - eval_results['predicted_return']

    # Save the evaluation results
    eval_results.to_csv('output/catboost_evaluation.csv', index=False)

    # Plot the feature importance
    import matplotlib.pyplot as plt
    import seaborn as sns

    feature_importance = model.get_feature_importance(prettified=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importances', y='Feature Id', data=feature_importance.head(10))
    plt.title('Top 10 Feature Importances')
    plt.show()

    return None
    