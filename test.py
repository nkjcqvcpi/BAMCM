import pandas as pd
import numpy as np

stock = pd.read_csv('第一届湾区数模赛题及数据/A题/附录一：30支股票行情.csv', header=0, index_col=1, parse_dates=['time'])
codes = stock['code'].unique()

trend = pd.DataFrame(index=codes)

trend[0] = np.zeros(30)

i= 0