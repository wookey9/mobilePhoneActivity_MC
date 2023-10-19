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

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

if os.path.exists("df_cdrs.pkl"):
    df_cdrs = pd.read_pickle("df_cdrs.pkl")

else:
    file_list = os.listdir('./input/')
    df_cdrs = pd.DataFrame({})
    for i,f in enumerate(sorted(file_list)):
        if 'sms-call-internet-mi-' in f:
            if i < 30:
                df = pd.read_csv('./input/' + f, parse_dates=['datetime'])

                df = df.fillna(0)
                df['sms'] = df['smsin'] + df['smsout']
                df['calls'] = df['callin'] + df['callout']
                print(df.head())

                df_cdrs_internet = df[['datetime', 'CellID', 'internet', 'calls', 'sms']].groupby(['datetime', 'CellID'], as_index=False).sum()
                df_cdrs_internet['hour'] = df_cdrs_internet.datetime.dt.hour+24*(df_cdrs_internet.datetime.dt.day-1) + 30*(df_cdrs_internet.datetime.dt.month - 11) * 24
                df_cdrs_internet['minute'] = df_cdrs_internet.datetime.dt.minute + 60*df_cdrs_internet.datetime.dt.hour+60*24*(df_cdrs_internet.datetime.dt.day-1) + 60*24*30*(df_cdrs_internet.datetime.dt.month - 11)
                df_cdrs_internet['groupId'] = ((df_cdrs_internet.CellID - 1) % 100 / 25).astype(int) + ((df_cdrs_internet.CellID - 1) / 100 / 25).astype(int) * 4 + 1
                df_cdrs_internet['sectorId'] = ((df_cdrs_internet.CellID - 1) % 100 / 5).astype(int) + ((df_cdrs_internet.CellID - 1) / 100 / 5).astype(int) * 20 + 1
                df_cdrs_internet = df_cdrs_internet.set_index(['minute']).sort_index()

                df_cdrs = pd.concat([df_cdrs, df_cdrs_internet])
                print(f)

    print('Data load done')
    df_cdrs.to_pickle("df_cdrs.pkl")

print('Data Frame Ready.')


# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

def create_sequences(data, seq_length):
    sequences = []
    target = []

    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append(seq)
        target.append(label)

    return np.array(sequences), np.array(target)

#target_cell_list = [4458,4459,4460,4558,4559,4560,4658,4659,4660]

target_cell_list = [7267,7268,7269,7367,7368,7369,7467,7468]

f2 = plt.figure()
df_merge = pd.DataFrame({})
for cell in target_cell_list:
    if len(df_merge) == 0:
        df_merge = df_cdrs[df_cdrs.CellID == cell]['internet']
    else:
        df_merge = pd.merge(df_merge, df_cdrs[df_cdrs.CellID == cell]['internet'], left_index=True, right_index=True, suffixes=("", str(cell)))

    df_cell = df_cdrs[df_cdrs.CellID == cell]['internet']
    y_ema = [df_cell.iloc[0], ]
    for y_i in df_cell.iloc[1:]:
        y_ema.append(y_ema[-1] * 0.9 + y_i * (1 - 0.9))
    plt.plot(df_cell.index - min(df_cell.index), y_ema, label=f'Cell {cell}')

df_merge.to_pickle('input_7267.pkl')
df_merge.to_csv('input_7267.csv')

plt.xlabel("Hour")
plt.ylabel("Number of connections")
plt.legend(loc='upper right')
sns.despine()

cellid = 0
f2 = plt.figure()
conerted_df_merge = (df_merge * 256).div(df_merge.sum(axis=1),axis=0)
for col,values in conerted_df_merge.items():
    y_ema = [values.iloc[0],]
    for y_i in values.iloc[1:]:
        y_ema.append(y_ema[-1]*0.9 + y_i*(1-0.9))
    plt.plot(values.index - min(values.index), y_ema, label=f'Cell {target_cell_list[cellid]}')
    cellid += 1


plt.show()
