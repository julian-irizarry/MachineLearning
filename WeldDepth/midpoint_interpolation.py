# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 13:49:56 2021

@author: Julian
"""

import pandas as pd

df = pd.read_csv("NN_Development.csv")
df = df.loc[:, ['Power \n(W)', 'Travel \nSpeed\n(mm/sec)',
                'Weld \nDepth\n(mm)']]
df2 = df.sort_values(by='Weld \nDepth\n(mm)')


def create_data(df):
    data = df.to_numpy()
    for i in range(98):
        index = 99 + i
        new_x = (data[i, 0] + data[i+1, 0]) / 2
        new_y = (data[i, 1] + data[i+1, 1]) / 2
        new_z = (data[i, 2] + data[i+1, 2]) / 2
        df2 = pd.DataFrame([[new_x, new_y, new_z]], columns=[
                           'Power \n(W)', 'Travel \nSpeed\n(mm/sec)', 'Weld \nDepth\n(mm)'], index=[index])
        df = df.append(df2)
    return df


df2 = create_data(df2).sort_index()
#df2.to_csv('interpolated_data.csv', index=False)
stats = df.describe()
stats2 = df2.describe()
