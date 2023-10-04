import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
import datetime
import os

file_list = os.listdir('./')
parse = lambda x: datetime.datetime.fromtimestamp(float(x)/1000)
for f in file_list:
    if '.txt' in f:
        df = pd.read_csv(f, sep='\t', encoding="utf-8-sig", names=['CellID', 'datetime', 'countrycode', 'smsin', 'smsout', 'callin', 'callout', 'internet'], parse_dates=['datetime'], date_parser=parse)
        csv_f = f.replace('txt','csv')
        df.to_csv(csv_f)
        print(f)