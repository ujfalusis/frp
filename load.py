from typing import Tuple
import pandas as pd
import datetime as dt

exchange = 'binance' # 'bybit'

bybitrf = pd.read_csv('fundingRatePrediction/bybitRealizedFundings.csv' if exchange == 'bybit' else 'fundingRatePrediction/binanceRealizedFundings.csv', index_col='timestamp', parse_dates=['timestamp'])
bybitrf.columns = bybitrf.columns.str.rstrip('_realizedFunding') # remove column names postfix
tspersymbol = bybitrf.apply(lambda x: bybitrf.index[~x.isna()].to_pydatetime(), axis = 0) # list timestamps for symbols
intervals = tspersymbol.apply(lambda x: list(set(x[1:] - x[:-1]))) # calculate intervals between timestamps for symbols
intervals
cycle8 = intervals[intervals.apply(lambda x: x == [dt.timedelta(hours=8)])].index # list symbols which have only 8 hours intervals
cycleother = intervals[intervals.apply(lambda x: x != [dt.timedelta(hours=8)])].index # list symbols which have other intervals than 8 hours
intervals[cycleother]

bybitrf = bybitrf[cycle8].dropna(axis = 0, how='all') # filter for rows where timestamp is 8 hours and symbol has only 8 hours interval 

bybitef = pd.read_csv('fundingRatePrediction/nextFundingRates.csv', usecols=['timestamp', 'exchange', 'symbol', 'nextFundingRate'])
masks = bybitef['symbol'].isin(cycle8)
maske = bybitef['exchange'] == ('BYBITU' if exchange == 'bybit' else 'BINANCEF') 
bybitef = bybitef[masks & maske].drop(columns=['exchange'])
bybitef['timestamp'] = bybitef['timestamp'].apply(lambda x: dt.datetime.fromtimestamp(x * 1e-3)) # filter for rows where exchange is ByBit and symbol has only 8 hours interval 

def estimationtsandremaining(ts: dt.datetime, cycle: dt.timedelta = dt.timedelta(hours=8)) -> Tuple[dt.datetime, dt.timedelta]:
    # ts = ts.replace(microsecond=0)
    secs = (ts - dt.datetime.min).total_seconds()
    cyclesecs = cycle.total_seconds()
    remainder = dt.timedelta(seconds=round(secs % cyclesecs, 3))
    return [(ts - remainder + cycle), cycle - remainder]

tsandremaining = bybitef['timestamp'].apply(lambda x: estimationtsandremaining(x.to_pydatetime()))
rem = pd.DataFrame(tsandremaining.to_list(), columns=['nextcyclets', 'remaining'])
bybitef = pd.concat([bybitef.reset_index(), rem], axis=1).drop('index', axis = 1)

bybitef['realized'] = bybitef.apply(lambda x: bybitrf.loc[x['nextcyclets'], x['symbol']], axis = 1)

def rmse(predictions, targets):
    return ((predictions - targets) ** 2).mean() ** 0.5

for i in range(1, 4):
    print(i)
    bybitef['lastcyclets' + str(i)] = bybitef['nextcyclets'] - dt.timedelta(hours=8 * i)
    bybitef['realizedbylast' + str(i)] = bybitef.apply(lambda x: bybitrf.loc[x['lastcyclets' + str(i)], x['symbol']], axis = 1)

rmse(bybitef['nextFundingRate'], bybitef['realized'])
rmse(bybitef['realizedbylast1'], bybitef['realized'])
rmse(bybitef['realizedbylast2'], bybitef['realized'])
rmse(bybitef['realizedbylast3'], bybitef['realized'])
rmse((bybitef['realizedbylast1'] + bybitef['realizedbylast2'] + bybitef['realizedbylast3']) / 3, bybitef['realized'])

bybitef['bin'] = bybitef['remaining'].apply(lambda x: int(x.total_seconds() / 60**2)) 
bybitef.groupby('bin')[['bin', 'realized', 'nextFundingRate']].apply(lambda x: rmse(x['realized'], x['nextFundingRate']))
bybitef.groupby('bin')[['bin', 'realized', 'realizedbylast1']].apply(lambda x: rmse(x['realized'], x['realizedbylast1']))
bybitef['realizedbycombined'] = bybitef.apply(lambda x: x['realizedbylast1'] if x['bin'] == 7 else x['nextFundingRate'], axis = 1) 
rmse(bybitef['realizedbycombined'], bybitef['realized'])
rmse(bybitef['nextFundingRate'], bybitef['realized'])

bybitef.to_pickle(f'./work_{exchange}.pck')
