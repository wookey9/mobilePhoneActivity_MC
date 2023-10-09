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
    df_cdrs_internet = pd.read_pickle("df_cdrs_internet.pkl")
    df_cdrs_group = pd.read_pickle("df_cdrs_group.pkl")
    df_cdrs_sector = pd.read_pickle("df_cdrs_sector.pkl")
else:
    file_list = os.listdir('./input/')
    df_cdrs = pd.DataFrame({})
    for f in file_list:
        if 'sms-call-internet-mi-' in f:
            df = pd.read_csv('./input/' + f, parse_dates=['datetime'])
            df_cdrs = pd.concat([df_cdrs, df])
            print(f)

    print('Data load done.')

    df_cdrs = df_cdrs.fillna(0)
    df_cdrs['sms'] = df_cdrs['smsin'] + df_cdrs['smsout']
    df_cdrs['calls'] = df_cdrs['callin'] + df_cdrs['callout']
    print(df_cdrs.head())

    df_cdrs_internet = df_cdrs[['datetime', 'CellID', 'internet', 'calls', 'sms']].groupby(['datetime', 'CellID'], as_index=False).sum()
    df_cdrs_internet['hour'] = df_cdrs_internet.datetime.dt.hour+24*(df_cdrs_internet.datetime.dt.day-1) + 30*(df_cdrs_internet.datetime.dt.month - 11) * 24
    df_cdrs_internet['groupId'] = ((df_cdrs_internet.CellID - 1) % 100 / 25).astype(int) + ((df_cdrs_internet.CellID - 1) / 100 / 25).astype(int) * 4 + 1
    df_cdrs_internet['sectorId'] = ((df_cdrs_internet.CellID - 1) % 100 / 5).astype(int) + ((df_cdrs_internet.CellID - 1) / 100 / 5).astype(int) * 20 + 1
    df_cdrs_internet = df_cdrs_internet.set_index(['hour']).sort_index()

    df_cdrs_group = df_cdrs[['datetime', 'CellID', 'internet', 'calls', 'sms']].groupby(['datetime', 'CellID'], as_index=False).sum()
    df_cdrs_group['hour'] = df_cdrs_group.datetime.dt.hour+24*(df_cdrs_group.datetime.dt.day-1)
    df_cdrs_group['groupId'] = ((df_cdrs_group.CellID - 1) % 100 / 25).astype(int) + ((df_cdrs_group.CellID - 1) / 100 / 25).astype(int) * 4 + 1
    df_cdrs_group['sectorId'] = ((df_cdrs_group.CellID - 1) % 100 / 5).astype(int) + ((df_cdrs_group.CellID - 1) / 100 / 5).astype(int) * 20 + 1

    df_cdrs_sector = df_cdrs_group[['sectorId','hour','internet','calls','sms']].groupby(['hour','sectorId'], as_index=False).sum()
    df_cdrs_sector = df_cdrs_sector.set_index(['hour']).sort_index()

    df_cdrs_group = df_cdrs_group[['groupId','hour','internet','calls','sms']].groupby(['hour','groupId'], as_index=False).sum()
    df_cdrs_group = df_cdrs_group.set_index(['hour']).sort_index()

    df_cdrs_sector['groupId'] = ((df_cdrs_sector.sectorId - 1) % 20 / 5).astype(int) + ((df_cdrs_sector.sectorId - 1) / 20 / 5).astype(int) * 4 + 1


    print(df_cdrs_internet)
    print(df_cdrs_group)

    conv_internet = []
    for g in range(1,17):
        conv_internet.append(np.array(df_cdrs_group[df_cdrs_group.groupId == g]['internet']).max())

    conv_internet = np.array(conv_internet)
    df_cdrs_sector['convInternet'] = df_cdrs_sector['internet'] * 256 / conv_internet[np.array(df_cdrs_sector.groupId) - 1]

    df_cdrs.to_pickle("df_cdrs.pkl")
    df_cdrs_internet.to_pickle("df_cdrs_internet.pkl")
    df_cdrs_group.to_pickle("df_cdrs_group.pkl")
    df_cdrs_sector.to_pickle("df_cdrs_sector.pkl")

print('Data Frame Ready.')
'''
f2 = plt.figure()
ax = (df_cdrs_group[df_cdrs_group.groupId == 1]['internet']).plot(label=f'Group: 1')
for g in range(2,17):
    (df_cdrs_group[df_cdrs_group.groupId == g]['internet']).plot(ax=ax, label=f'Group: {g}')

plt.xlabel("Weekly hour")
plt.ylabel("Number of connections")
plt.legend(loc='best')
sns.despine()

target_group = 11
sectorList = []
for i,g in df_cdrs_internet.iterrows():
    if g.groupId == target_group:
        if g.sectorId not in sectorList:
            sectorList.append(g.sectorId)

cnt = 0
f2 = plt.figure()
for sec in sectorList:
    if cnt == 0:
        ax = (df_cdrs_sector[df_cdrs_sector.sectorId == sec]['convInternet']).plot(label=f'Sector: {sec}')
    else:
        (df_cdrs_sector[df_cdrs_sector.sectorId==sec]['convInternet']).plot(ax=ax, label=f'Sector {sec}')
    cnt += 1

plt.xlabel("Weekly hour")
plt.ylabel("Number of connections")
plt.legend(loc='best')
sns.despine()


f2 = plt.figure()
ax = (df_cdrs_sector[df_cdrs_sector.sectorId == 211]['convInternet']).plot(label=f'Sector {211}')
df_cdrs_sector[df_cdrs_sector.sectorId==212]['convInternet'].plot(ax=ax, label=f'Sector {212}')
df_cdrs_sector[df_cdrs_sector.sectorId==213]['convInternet'].plot(ax=ax, label=f'Sector {213}')
df_cdrs_sector[df_cdrs_sector.sectorId==231]['convInternet'].plot(ax=ax, label=f'Sector {231}')
df_cdrs_sector[df_cdrs_sector.sectorId==232]['convInternet'].plot(ax=ax, label=f'Sector {232}')
df_cdrs_sector[df_cdrs_sector.sectorId==233]['convInternet'].plot(ax=ax, label=f'Sector {233}')
df_cdrs_sector[df_cdrs_sector.sectorId==251]['convInternet'].plot(ax=ax, label=f'Sector {251}')
df_cdrs_sector[df_cdrs_sector.sectorId==252]['convInternet'].plot(ax=ax, label=f'Sector {252}')
df_cdrs_sector[df_cdrs_sector.sectorId==253]['convInternet'].plot(ax=ax, label=f'Sector {253}')

plt.xlabel("Weekly hour")
plt.ylabel("Number of connections")
plt.legend(loc='best')
sns.despine()


plt.show()
'''


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

target_cell_list = [211,213,231,232,233,251,252,253]
model_list = []


for sec in target_cell_list:
    # Sample data: A sine wave
    y = df_cdrs_sector[df_cdrs_sector.sectorId == sec]['convInternet'].values

    # Create sequences from the sine wave
    seq_length = 5

    x, y = create_sequences(y, seq_length)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(2)
    y = torch.tensor(y, dtype=torch.float32)

    # Split data into train and test sets
    train_size = int(0.8 * len(x))  # 80% of data for training
    test_size = len(x) - train_size

    x_train, x_test = torch.split(x, [train_size, test_size])
    y_train, y_test = torch.split(y, [train_size, test_size])

    # Hyperparameters
    input_dim = 1
    hidden_dim = 50
    num_layers = 1
    output_dim = 1

    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim, )

    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    # Train the model
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        outputs = model(x_train)
        optimizer.zero_grad()

        # Obtain the loss function
        loss = criterion(outputs, y_train.unsqueeze(1))
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    # Predict
    model.eval()
    with torch.no_grad():
        predictions = model(x_test)

    model_list.append(model)

    # Convert predictions to a numpy array or pandas series for further analysis
    predictions = predictions.squeeze(1).numpy()

    print(predictions)


    f2 = plt.figure()
    plt.plot(y_test, label=f'Sector: {sec}')
    plt.plot(predictions, label='prediction')
    plt.xlabel("Weekly hour")
    plt.ylabel("Number of connections")
    plt.legend(loc='best')
    sns.despine()

for i, m in enumerate(model_list):
    torch.save(m, 'lstm_model_' + str(i))

f2 = plt.figure()
df_merge = pd.DataFrame({})
for sec in target_cell_list:
    if len(df_merge) == 0:
        df_merge = df_cdrs_sector[df_cdrs_sector.sectorId == sec]['internet']
    else:
        df_merge = pd.merge(df_merge, df_cdrs_sector[df_cdrs_sector.sectorId == sec]['internet'], left_index=True, right_index=True, suffixes=("",str(sec)))

    plt.plot(df_cdrs_sector[df_cdrs_sector.sectorId == sec]['convInternet'], label=f'Sector {sec}')

df_merge.to_pickle('input.pkl')

plt.xlabel("Weekly hour")
plt.ylabel("Number of connections")
plt.legend(loc='best')
sns.despine()


plt.show()
