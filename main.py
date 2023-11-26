import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image

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
            df['minute'] = df.datetime.dt.minute + df.datetime.dt.hour*60 + (df.datetime.dt.day)*24*60 + (df.datetime.dt.month - 11) * 60*24*30
            df = df[['minute','CellID','internet', 'smsin','smsout', 'callin','callout']].groupby(['minute', 'CellID'], as_index=False).sum()
            df_all = pd.concat([df_all, df])
            print(f)
    df_all = df_all.set_index('minute').sort_index()
    print(df_all.head())
    df_all.to_csv('dataset.csv')

if not os.path.exists("up_ratio.csv"):
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

    df_out_ratio = (df_out * 100 / (df_in + df_out))
    df_out_ratio.to_csv('up_ratio.csv')
else:
    df_out_ratio = pd.read_csv("up_ratio.csv")

df_out_ratio['0'].plot()
plt.show()

df = df_out_ratio
# row = 24
# column = 15 ## IDS, Pangyo 데이터셋은 위 사이즈로 진행
row = 20
column = 15
print(df.shape[0])
print(df.max(axis=0)[1])
print(df.min(axis=0)[1])
for a in range(int(df.shape[0] / (row*4))):
    image = Image.new("L", (row, column))
    im = image.load()
    (width, height) = image.size
    maxval = df.max(axis=0)[1]
    minval = df.min(axis=0)[1]
    for i in range(0, height):
        for j in range(0, width):
            color = int(((df.values[(a * row * 4) + i * row + j][1]) - minval) / (maxval - minval)*255)
            im[j, i] = (255-color)
    imarr1 = np.asarray(image)
    print(imarr1)
    # filename = "./Dataset/Data_Pangyo_2415/image_sliding/txt/Pangyo_2020_" + str(a) + ".txt"
    filename = "./image_sliding/txt/MILAN_UP_RATIO_" + str(a) + ".txt"
    # filename_ = "./Dataset/qwer4/Pangyo_2020_" + str(a)+"_.txt"
    # filename1 = "./Dataset/Data_Pangyo_2415/image_sliding/Pangyo_2020_" + str(a)+".png"
    filename1 = "./image_sliding/MILAN_UP_RATIO_" + str(a)+".png"
    image.save(filename1)
    np.savetxt(filename, imarr1, fmt='%d')
    current_image = Image.open(filename1)