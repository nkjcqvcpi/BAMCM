import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

HANDLING_FEE = 2.5e-4

stock = pd.read_csv('第一届湾区数模赛题及数据/A题/附录一：30支股票行情.csv', header=0, index_col=1, parse_dates=['time'])
codes = stock['code'].unique()

stock_g = stock.groupby("code")

gb_index = pd.read_csv('第一届湾区数模赛题及数据/A题/附录二：大湾区指数行情.csv', header=0, index_col=1, parse_dates=['time'])
trend = []

for code in tqdm(codes):
    pers = []
    bias = stock_g.get_group(code)['close']/stock_g.get_group(code)['open'] - 1
    trend.append(bias.pct_change(periods=30))


trend = pd.DataFrame(trend)
trend.index = codes

timeindex = list(stock_g.get_group(codes[0])['open'].resample('W').groups.keys())

bp = trend[trend.columns[trend.columns.weekday == 1]].replace([np.inf, -np.inf], np.nan).fillna(0)
sp = trend[trend.columns[trend.columns.weekday == 0]].replace([np.inf, -np.inf], np.nan).fillna(0)

investment_per = pd.DataFrame(np.zeros([codes.shape[0], trend.shape[1]]))
investment_per.index = codes.T
week_top10 = []

for i in tqdm(range(sp.shape[1])):
    try:
        t = pd.DataFrame.sort_values(sp, sp.columns[i])[sp.columns[i]][:10]
        week_top10.append(t.index)
        p_tmp = t[t > 0]
        if p_tmp.shape[0]:
            pp = 1 - 0.01 * p_tmp.shape[0]
            for code in t.index:
                if trend.at[code, i] > 0:
                    investment_per.at[code, i] = (p_tmp[code] / p_tmp.sum()) * pp
                else:
                    investment_per.at[code, i] = 0.01
        else:
            n_tmp = (pd.Series.sort_values(t)).abs()
            n_tmp.index = n_tmp.index[::-1]
            for code in t.index:
                investment_per.at[code, i] = (n_tmp[code] / n_tmp.sum())
        investment_per[i].fillna(0)
    except Exception:
        pass

principal = np.zeros(bp.shape[1])
principal[0] = 1

buy_p = (stock[stock.index.weekday == 1])[['code', 'open']].groupby('code')[['open']]
sell_p = (stock[stock.index.weekday == 0])[['code', 'close']].groupby('code')[['close']]

for n, week in tqdm(enumerate(week_top10[:-1])):
    for sc in week:
        try:
            sp = sell_p.get_group(sc).reset_index(drop=True)
            bp = buy_p.get_group(sc).reset_index(drop=True)
            principal[n+1] += ((principal[n] * 0.1)/sp.at[n, 'close']) * bp.at[n, 'open']  # investment_per.at[sc, n]
        except Exception:
            principal[n+1] = principal[n]
            break

yield_curve = pd.Series(principal).fillna(0).pct_change(periods=24)

gb_principal = np.zeros(481)
gb_principal[0] = 1

buy_p = gb_index[gb_index.index.weekday == 1].resample('W')['open']
sell_p = gb_index[gb_index.index.weekday == 0].resample('W')['close']

for week in tqdm(range(480)):
    try:
        gb_principal[week+1] = gb_principal[week]*sell_p.get_group(timeindex[week+1])[0] / buy_p.get_group(timeindex[week])[0]
    except Exception:
        gb_principal[week + 1] = gb_principal[week]


gb_yield_curve = pd.Series(gb_principal).fillna(0).pct_change(periods=24)

plt.plot(range(481), yield_curve, color='green')
plt.plot(range(481), gb_yield_curve, color='red')
plt.show()

i = 0