import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os

sns.set_style("ticks")
sns.set_context("paper")

file_list = os.listdir('./input/')
target_cell_list = [4458,4459,4460,4558,4559,4560,4658,4659,4660]
df_all = pd.DataFrame({})

if os.path.exists("dataset.csv"):
    df_all = pd.read_csv("dataset.csv")
    df_all = df_all.set_index('minute').sort_index()
    print(df_all.head())
else:
    for i,f in enumerate(sorted(file_list)):
        if 'sms-call-internet-mi-' in f:
            df = pd.read_csv('./input/' + f, parse_dates=['datetime'])
            df = df.fillna(0)
            df['minute'] = df.datetime.dt.minute + df.datetime.dt.hour*60 + (df.datetime.dt.day - 1)*24*60 + (df.datetime.dt.month - 11) * 60*24*30
            df = df[['minute','CellID','internet', 'smsin','smsout', 'callin','callout']].groupby(['minute', 'CellID'], as_index=False).sum()
            df_all = pd.concat([df_all, df])
            print(f)
    df_all = df_all.set_index('minute').sort_index()
    print(df_all.head())
    df_all.to_csv('dataset.csv')

df_in = pd.DataFrame({})
df_out = pd.DataFrame({})
for cell in target_cell_list:
    df = df_all[df_all.CellID == cell]['smsin'] + df_all[df_all.CellID == cell]['callin']
    if len(df_in) == 0:
        df_in = df
    else:
        df_in = df_in + df

for cell in target_cell_list:
    df = df_all[df_all.CellID == cell]['smsout'] + df_all[df_all.CellID == cell]['callout']
    if len(df_out) == 0:
        df_out = df
    else:
        df_out = df_out + df

df_in.plot()
df_out.plot()

f = plt.figure()
df_out_ratio = (df_out * 100 / (df_in + df_out))

y_ema = [df_out_ratio.iloc[0], ]
for y_i in df_out_ratio.iloc[1:]:
    y_ema.append(y_ema[-1] * 0.9 + y_i * (1 - 0.9))
plt.plot(y_ema)

plt.show()