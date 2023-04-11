import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

kp = pd.read_csv('kospi.csv')
nd = pd.read_csv('nasdaq.csv')
ts = pd.read_excel('trans.xlsx')


#코스피 정규화
kp_mean = np.mean(kp, axis=0)
kp_std = np.std(kp, axis=0)

kp_scaled = (kp - kp_mean)/kp_std

#나스닥 정규화
nd_mean = np.mean(nd, axis=0)
nd_std = np.std(nd, axis=0)

nd_scaled = (nd - nd_mean)/nd_std

#환율 정규화
ts_mean = np.mean(ts, axis=0)
ts_std = np.std(ts, axis=0)

ts_scaled = (ts - ts_mean)/ts_std

from sklearn.linear_model import LinearRegression

#코스피 & 나스닥 선형 회귀
train_input, test_input, train_target, test_target = train_test_split(kp_scaled, nd_scaled, test_size=0.3, random_state=42)

lr = LinearRegression()
lr.fit(train_input, train_target)
print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))

print('\n')

#코스피 & 환율 선형 회귀
train_input, test_input, train_target, test_target = train_test_split(kp_scaled, ts_scaled, test_size=0.3, random_state=42)

lr = LinearRegression()
lr.fit(train_input, train_target)
print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))

print('\n')

#나스닥 & 환율 선형 회귀
train_input, test_input, train_target, test_target = train_test_split(nd_scaled, ts_scaled, test_size=0.3, random_state=42)

lr = LinearRegression()
lr.fit(train_input, train_target)
print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))

print('\n')

#다중 회귀 (코스피)
X = pd.concat([pd.DataFrame(kp_scaled), pd.DataFrame(nd_scaled), pd.DataFrame(ts_scaled)], axis=1)
train_input, test_input, train_target, test_target = train_test_split(X, kp_scaled, test_size=0.3, random_state=42)
lr.fit(train_input, train_target)
print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))