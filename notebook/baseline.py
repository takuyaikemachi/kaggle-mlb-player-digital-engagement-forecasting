#%%
import numpy as np
import pandas as pd
import gc
import pickle
import os
import datetime as dt
import matplotlib.pyplot as plt
import lightgbm as lbg
from sklearn.metrics import mean_absolute_error
import warnings
warnings.simplefilter('ignore')
pd.options.display.float_format = '{:10.4f}'.format

#%%
df_train_source = pd.read_csv('../input/train_updated.csv')
print(df_train_source.shape)
display(df_train_source.head())

#%%
df_train_source = df_train_source.loc[df_train_source['date']>=20200401, :].reset_index(drop=True)
print (df_train_source.shape)

#%%
def unpack_json(json_str):
    return np.nan if pd.isna(json_str) else pd.read_json(json_str)

def extract_data(input_df, col='events', show=False):
    output_df = pd.DataFrame()
    for i in np.arange(len(input_df)):
        if show: print('\r{}/{}'.format(i+1, len(input_df)), end='')
        try:
            output_df = pd.concat([
                output_df, unpack_json(input_df[col].iloc[i])
            ], axis=0, ignore_index=True)
        except:
            pass
    if show: print('')
    if show: print(output_df.shape)
    if show: display(output_df.head())
    return output_df

#%%
df_engagement = extract_data(df_train_source, col='nextDayPlayerEngagement', show=True)

#%%
df_engagement['date_playerId'] = df_engagement['engagementMetricsDate'].str.replace('-', '') + '_' + df_engagement['playerId'].astype(str)
df_engagement.head()

#%%
df_engagement['date'] = pd.to_datetime(df_engagement['engagementMetricsDate'], format='%Y-%m-%d') + dt.timedelta(days=-1)

df_engagement['dayofweek'] = df_engagement['date'].dt.dayofweek
df_engagement['yearmonth'] = df_engagement['date'].astype(str).apply(lambda x: x[:7])
df_engagement.head()

#%%
df_players = pd.read_csv('../input/players.csv')
print(df_players.shape)
print(df_players['playerId'].agg('nunique'))
df_players.head()

#%%
df_players['playerForTestSetAndFuturePreds'] = np.where(df_players['playerForTestSetAndFuturePreds']==True, 1, 0)
print(df_players['playerForTestSetAndFuturePreds'].sum())
print(df_players['playerForTestSetAndFuturePreds'].mean())

#%%
df_train = pd.merge(df_engagement, df_players, on=['playerId'], how='left')
print(df_train.shape)
df_train.head()

#%%
x_train = df_train[[
    'playerId', 'dayofweek', 'birthCity', 'birthStateProvince', 'birthCountry', 'heightInches', 'weight', 'primaryPositionCode', 'primaryPositionName', 'playerForTestSetAndFuturePreds'
]]
y_train = df_train[['target1', 'target2', 'target3', 'target4']]
id_train = df_train[['engagementMetricsDate', 'playerId', 'date_playerId', 'date', 'yearmonth', 'playerForTestSetAndFuturePreds']]
print(x_train.shape, y_train.shape, id_train.shape)
x_train.head()

#%%
for col in ['playerId', 'dayofweek', 'birthCity', 'birthStateProvince', 'birthCountry', 'primaryPositionCode', 'primaryPositionName']:
    x_train[col] = x_train[col].astype('category')

#%%
list_cv_month = []
for [list_2020, list_2021, list_final] in [
    [[5,6,7,8,9,10,11,12],[1,2,3,4],['2021-05']],
    [[6,7,8,9,10,11,12],[1,2,3,4,5],['2021-06']],
    [[7,8,9,10,11,12],[1,2,3,4,5,6],['2021-07']]]:
    list_output = ['2020-{:0=2}'.format(i) for i in list_2020]
    list_output.extend(['2021-{:0=2}'.format(i) for i in list_2021])
    list_output = [list_output, list_final]
    list_cv_month.append(list_output)
print(list_cv_month)

#%%
cv = []
for month_tr, month_va in list_cv_month:
    cv.append([
        id_train.index[id_train['yearmonth'].isin(month_tr)],
        id_train.index[id_train['yearmonth'].isin(month_va) & id_train['playerForTestSetAndFuturePreds']==1]
    ])

cv[0]