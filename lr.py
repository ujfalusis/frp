import numpy as np
import statsmodels.api as sm
import pandas as pd
import datetime as dt


def rmse(predictions, targets):
    return ((predictions - targets) ** 2).mean() ** 0.5

bybitef = pd.read_pickle('./work.pck')
bybitef.columns
mask = ~bybitef['realizedbylast1'].isna()
bybitef = bybitef[mask]

bybitef_cp = bybitef.copy()

remaining = bybitef['remaining'].apply(lambda x: pd.Timedelta(x).to_pytimedelta().total_seconds())
rem = np.cos(remaining * 2 * np.pi / dt.timedelta(hours=8).total_seconds())
rem.shape
rem = np.stack([rem, np.sin(remaining * 2 * np.pi / dt.timedelta(hours=8).total_seconds())], axis=1)
rem.shape
bybitef = bybitef[['nextFundingRate', 'realizedbylast1', 'realized']]

x = bybitef[['nextFundingRate', 'realizedbylast1']].to_numpy()
x = np.concatenate([x, rem], axis = 1)
y = bybitef['realized'].to_numpy()
x.shape
y.shape
x = sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()

results.summary()

result = results.predict(x)

rmse(bybitef['nextFundingRate'], bybitef['realized'])
rmse(result, bybitef['realized'])
rmse(bybitef_cp['realizedbycombined'], bybitef['realized'])


0.0006672364144507439 / 0.0007622358591199435

mask = bybitef.isna().any(axis = 1)



bybitef.drop()