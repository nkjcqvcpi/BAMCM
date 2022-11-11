import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

HANDLING_FEE = 2.5e-4

stock = pd.read_csv('第一届湾区数模赛题及数据/A题/附录一：30支股票行情.csv', header=0, index_col=1, parse_dates=['time'])
codes = stock['code'].unique()

print(codes)

stock_g = stock.groupby("code")

timeindex = list(stock_g.get_group(codes[0])['open'].resample('W').groups.keys())

gb_index = pd.read_csv('第一届湾区数模赛题及数据/A题/附录二：大湾区指数行情.csv', header=0, index_col=1, parse_dates=['time'])

trend = []

for code in tqdm(codes):
    pers = []
    op = stock_g.get_group(code)['open'].resample('W')
    cl = stock_g.get_group(code)['close'].resample('W')
    for week in timeindex:
        try:
            bias = cl.get_group(week)/op.get_group(week) - 1
            pers.append(bias.mean())
        except KeyError:
            pers.append(pers[-1])
    trend.append(pers)

trend = pd.DataFrame(trend)
week_top10 = []
for i in tqdm(range(trend.shape[1])):
    week_top10.append(pd.DataFrame.sort_values(trend, i).index[:10])

principal = np.zeros(trend.shape[1])
principal[0] = 1

buy_p = stock[stock.index.weekday == 1].resample('W')['code', 'open']
sell_p = stock[stock.index.weekday == 0].resample('W')['code', 'close']

for n, week in tqdm(enumerate(week_top10[:-1])):
    try:
        bp = buy_p.get_group(timeindex[n])
        bp.index = bp['code']
        bp = bp['open']
        sp = sell_p.get_group(timeindex[n+1])
        sp.index = sp['code']
        sp = sp['close']
    except Exception:
        principal[n + 1] = principal[n]
    else:
        for sc in week:
            try:
                principal[n+1] += ((principal[n] * 0.1)/sp[codes[sc]]) * bp[codes[sc]]
            except Exception:
                principal[n+1] = principal[n]
                break

yield_curve = pd.Series(principal).pct_change(periods=12)


gb_principal = np.zeros(trend.shape[1])
gb_principal[0] = 1

buy_p = gb_index[gb_index.index.weekday == 1].resample('W')['open']
sell_p = gb_index[gb_index.index.weekday == 0].resample('W')['close']

for week in tqdm(range(trend.shape[1]-1)):
    try:
        gb_principal[week+1] = gb_principal[week]*sell_p.get_group(timeindex[week+1])[0] / buy_p.get_group(timeindex[week])[0]
    except Exception:
        gb_principal[week + 1] = gb_principal[week]


gb_yield_curve = pd.Series(gb_principal).pct_change(periods=12)

plt.plot(range(513), yield_curve, color='green')
plt.plot(range(513), gb_yield_curve, color='red')
plt.show()

i = 0