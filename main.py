import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

kp = pd.read_csv('kospi.csv')
nd = pd.read_csv('nasdaq.csv')
ex = pd.read_excel('trans.xlsx')


#코스피 정규화
kp_mean = np.mean(kp, axis=0)
kp_std = np.std(kp, axis=0)

kp_scaled = (kp - kp_mean)/kp_std

#나스닥 정규화
nd_mean = np.mean(nd, axis=0)
nd_std = np.std(nd, axis=0)

nd_scaled = (nd - nd_mean)/nd_std

#환율 정규화
ex_mean = np.mean(ex, axis=0)
ex_std = np.std(ex, axis=0)

ex_scaled = (ex - ex_mean)/ex_std

from sklearn.linear_model import LinearRegression

print('코스피 & 나스닥 선형 회귀')
#코스피 & 나스닥 선형 회귀
train_input, test_input, train_target, test_target = train_test_split(kp_scaled, nd_scaled, test_size=0.3, random_state=42)

lr = LinearRegression()
lr.fit(train_input, train_target)
print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))

print('\n코스피 & 환율 선형 회귀')

#코스피 & 환율 선형 회귀
train_input, test_input, train_target, test_target = train_test_split(kp_scaled, ex_scaled, test_size=0.3, random_state=42)

lr = LinearRegression()
lr.fit(train_input, train_target)
print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))

print('\n나스닥 & 환율 선형 회귀')

#나스닥 & 환율 선형 회귀
train_input, test_input, train_target, test_target = train_test_split(nd_scaled, ex_scaled, test_size=0.3, random_state=42)

lr = LinearRegression()
lr.fit(train_input, train_target)
print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))


from sklearn.linear_model import Ridge
print('\n코스피 & 나스닥 릿지')

#코스피 & 나스닥 릿지
train_input, test_input, train_target, test_target = train_test_split(kp_scaled, nd_scaled, test_size=0.3, random_state=42)

rd = Ridge(alpha=1)
rd.fit(train_input, train_target)
print(rd.score(train_input, train_target))
print(rd.score(test_input, test_target))

print('\n코스피 & 환율 릿지')

#코스피 & 환율 릿지
train_input, test_input, train_target, test_target = train_test_split(kp_scaled, ex_scaled, test_size=0.3, random_state=42)

rd = Ridge(alpha=1)
rd.fit(train_input, train_target)
print(rd.score(train_input, train_target))
print(rd.score(test_input, test_target))

print('\n나스닥 & 환율 릿지')

#나스닥 & 환율 릿지
train_input, test_input, train_target, test_target = train_test_split(nd_scaled, ex_scaled, test_size=0.3, random_state=42)

rd = Ridge(alpha=1)
rd.fit(train_input, train_target)
print(rd.score(train_input, train_target))
print(rd.score(test_input, test_target))

from sklearn.linear_model import Lasso
print('\n코스피 & 나스닥 라쏘')

#코스피 & 나스닥 라쏘
train_input, test_input, train_target, test_target = train_test_split(kp_scaled, nd_scaled, test_size=0.3, random_state=42)

ls = Lasso(alpha=0.01)
ls.fit(train_input, train_target)
print(ls.score(train_input, train_target))
print(ls.score(test_input, test_target))

print('\n코스피 & 환율 라쏘')

#코스피 & 환율 라쏘
train_input, test_input, train_target, test_target = train_test_split(kp_scaled, ex_scaled, test_size=0.3, random_state=42)

ls = Lasso(alpha=0.01)
ls.fit(train_input, train_target)
print(ls.score(train_input, train_target))
print(ls.score(test_input, test_target))

print('\n나스닥 & 환율 라쏘')

#나스닥 & 환율 라쏘
train_input, test_input, train_target, test_target = train_test_split(nd_scaled, ex_scaled, test_size=0.3, random_state=42)

ls = Lasso(alpha=0.01)
ls.fit(train_input, train_target)
print(ls.score(train_input, train_target))
print(ls.score(test_input, test_target))