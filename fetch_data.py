# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:55:30 2021

@author: JasonJhan
"""

import requests
import pandas as pd
import random
import xlwings as xw
import time
import json
import numpy as np


def fetch(url):
    res = requests.get(url)
    data = res.json()
    
    return data["data"]


start = "20121201"
stop = "20211202" #108/12/1~2無資料
date_list = pd.date_range(start, stop, freq='M').strftime("%Y%m%d").tolist()
TAIEX_df = pd.DataFrame([], columns=["date", "open", "high", "low", "close", "volume"])


for month in date_list:
    print(month)
    t = random.randint(5,20)
    time.sleep(t)
    url1 = f"https://www.twse.com.tw/indicesReport/MI_5MINS_HIST??response=json&date={month}&type=ALL"
    url2 = f"https://www.twse.com.tw/exchangeReport/FMTQIK?response=json&date={month}&type=ALL"
    data1 = fetch(url1)
    data2 = fetch(url2)
    for index in range(len(data1)):
        temp = {"date":data1[index][0], "open":data1[index][1], "high":data1[index][2], "low":data1[index][3], "close":data1[index][4], "volume":data2[index][1]}
        TAIEX_df = TAIEX_df.append(temp,  ignore_index=True)

# TAIEX_df.to_csv("TAIEX.csv")


#%% yfinance
import yfinance as yf 
TAI = yf.download('^TWII',start='2012-12-01',end='2021-12-02')
TAI.to_csv("TAIEX_fetch.csv")