import pandas as pd
import numpy as np
import statsmodels.api as sm

# Asset Pricing

# Load data and rename variables
df_returns = pd.read_csv('data/top_10.csv')
df_returns.columns = ['permno', 'date', 'name', 'return', 'market']

# List of identifiers (permno) from the given file
permno_df = pd.read_csv('data/permno-hw-2.csv')
permno = list(permno_df['permno'])

# Initialize CAPM betas dataframe and calculate betas
betas = pd.DataFrame(permno, columns=['permno']) 
betas['beta'] = np.nan
betas['alphaM'] = np.nan
betas['betaM'] = np.nan
betas['company_name'] = ""

# Loop through stocks to estimate CAPM beta for each
for i in permno:
    
    # Extract sub-dataframe
    subset = df_returns[df_returns['permno'] == i].dropna()
    
    # Store company name
    if not subset.empty:
        betas.loc[betas['permno'] == i, 'company_name'] = subset['name'].iloc[0]
    
    # Prepare for regression
    x0 = np.array(subset['market'])
    x1 = sm.add_constant(np.array(subset['market']))
    y = np.array(subset['return'])
    
    # Estimate CAPM
    capm = sm.OLS(y, x0).fit()
    
    # Estimate Market Model
    mktModel = sm.OLS(y, x1).fit()
    
    # Store parameters
    betas.loc[betas['permno'] == i, 'beta'] = capm.params[0]
    betas.loc[betas['permno'] == i, 'alphaM'] = mktModel.params[0]
    betas.loc[betas['permno'] == i, 'betaM'] = mktModel.params[1]

# Calculate CAPM-implied expected return using 7% market return
betas['expected_return'] = betas['beta'] * 7  # 7% is the long-run, inflation-adjusted market return

# Find Acadia Pharmaceuticals expected return
acadia_return = betas.loc[betas['permno'] == 90177, 'expected_return'].values[0]
acadia_beta = betas.loc[betas['permno'] == 90177, 'beta'].values[0]

print(f"Expected return for Acadia Pharmaceuticals: {acadia_return:.2f}%")
print(f"Beta for Acadia Pharmaceuticals: {acadia_beta:.2f}")

# Print all company betas and expected returns
print("\nBetas and Expected Returns for San Diego Companies:")
for _, row in betas.iterrows():
    print(f"{row['company_name']}: Beta = {row['beta']:.2f}, Expected Return = {row['expected_return']:.2f}%")
