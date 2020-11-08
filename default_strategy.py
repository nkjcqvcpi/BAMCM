import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import datetime

PRINCIPAL = 1
HANDLING_FEE = 2.5e-4
buy_d = []
sell_d = []


def date_init():
    day = datetime.datetime(2011, 1, 4, 15, 00, 00)
    end_day = datetime.datetime(2020, 10, 30, 15, 00, 00)
    delta = datetime.timedelta(days=1)
    flag = 1
    while day <= end_day:
        if day.weekday() == 0:
            sell_d.append(day)
            flag = 0
        elif day.weekday() == 1 and flag == 0:
            buy_d.append(day)
        day += delta


def judge_per_week(sbd):
    vari = []
    pers = []
    begin = 0
    end = 0
    flag = 1
    while end < len(sbd) - 4:
        while sbd.index[end].weekday() < sbd.index[end + 1].weekday():
            end += 1
        end += 1
        if sbd.index[begin].weekday() == 0:
            if (flag == 0) and (sbd.index[begin+1] not in buy_d):
                buy_d.append(sbd.index[begin+1])
            elif sbd.index[begin] not in sell_d:
                sell_d.append(sbd.index[begin])
                flag = 0
        op = sbd['open'][begin:end]
        cl = sbd['close'][begin:end]
        vari.append(np.std(sbd['amount'][begin:end] / sbd['volume'][begin:end]))
        begin = end
        per = cl / op - 1
        pers.append(per.mean())
    return pers, vari


stock = pd.read_csv('第一届湾区数模赛题及数据/A题/附录一：30支股票行情.csv', header=0, index_col=[2, 1], parse_dates=['time'])
codes = stock.index.get_level_values(0).unique()
rst = pd.date_range(start="2011-01-04 15:00:00", end="2020-10-30 15:00:00")
for code in tqdm(codes):
    eval(str(code.replace('.', ''))+" = stock.loc[code].reindex(rst, fill_value=0)")


gb_index = pd.read_csv('第一届湾区数模赛题及数据/A题/附录二：大湾区指数行情.csv', header=0, index_col=1, parse_dates=['time'])

investment_per = pd.DataFrame(np.ones(codes.shape[0]) * 0.1)
investment_per.index = codes.T
trend = []

for i, c in tqdm(enumerate(codes)):
    code = stock_g.get_group(c)
    percent, var = judge_per_week(code)
    trend.append(percent)

sell_d = pd.to_datetime(sell_d)
buy_d = pd.to_datetime(buy_d)

impcodeb = []
impcodes = []
imp_b = stock.loc[buy_d, ['code', 'open']]
imp_s = stock.loc[sell_d, ['code', 'close']]
for name, group in imp_b.groupby('code'):
    impcodeb.append(pd.Series(group['open'], index=group.index, name=name))

for name, group in imp_s.groupby('code'):
    impcodes.append(pd.Series(group['close'], index=group.index, name=name))

imp_b = pd.DataFrame(impcodeb)
imp_s = pd.DataFrame(impcodes)

trend = pd.DataFrame(trend)
trend.index = codes.T

week_top10 = []
for i in tqdm(range(trend.shape[1])):
    week_top10.append(pd.DataFrame.sort_values(trend, i).index[:10])

principal = np.zeros(trend.shape[1] + 1)
principal[0] = 1


def default_strategy():
    for n, week in tqdm(enumerate(week_top10)):
        for sc in week:
            principal[n + 1] += ((principal[n] * investment_per.at[sc, 0]) / imp_b.at[sc, buy_d[n]]) * imp_s.at[sc, sell_d[n]]


default_strategy()

gb_position = pd.DataFrame(np.zeros(buy_d.__len__()))
gb_principal = np.zeros(trend.shape[1] + 1)
gb_principal[0] = 1


def greater_bay_strategy():
    for n in tqdm(range(trend.shape[1])):
        gb_position[0][n] = gb_principal[n] / gb_index.at[buy_d[n], 'open']
        gb_principal[n + 1] = gb_position[0][n] * gb_index.at[sell_d[n], 'close']


greater_bay_strategy()

plt.scatter(trend.columns, principal[1:], color='green')
plt.scatter(trend.columns, gb_principal[1:], color='red')
plt.show()
