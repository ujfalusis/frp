import pandas as pd
import matplotlib.pyplot as plt

exchange = 'binance' # 'bybit'

bybitef = pd.read_pickle(f'./work_{exchange}.csv')
bybitef.columns
bybitef['diff'] = bybitef['realized'] - bybitef['nextFundingRate']
# bp = bybitef.boxplot(column='diff', by='bin', showfliers=True, whis = 100)
# bybitef.hist(column='diff', bins=1000, log = True, range = (-4e-2, 4e-2), cumulative  = False)

plt.show()

