---
title: 'Fintech COM HW3 詹凱錞'
disqus: hackmd
---

Fintech COM HW3 詹凱錞
===

Some modules are required
please install them use
```
>> pip install -r requirements.txt
```

## Fetch TAIEX data
Run the code which is in fetch_data.py.  
There are two ways to fetch the data.  

* One is crawl the data from https://www.twse.com.tw/zh/ (The main website of Taiwan Stock Exchange) using requests.  
* The more efficient way is using yfinance api to get data from https://tw.stock.yahoo.com/  




## HW2 main
### initial  
The HW2 code is in hw2_main.py and import some function in hw2_models.py  
Before analysing the problems, we should import some modules and read the csv we fetched just.
```python
import pandas as pd
import numpy as np

from hw3_models import triple_barrier, bios_MA, RSI
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from itertools import cycle

TAIEX_df = pd.read_csv("TAIEX_fetch.csv")
TAIEX_df = TAIEX_df.drop(columns=["Adj Close"])
```
### Question1B : Create triple-barrier labels
```python
# In[1B]
ret = triple_barrier(TAIEX_df.Close, 1.04, 0.98, 20) #use funciton triple_barrier in hw3_models
TAIEX_df["TB_label"] = ret.triple_barrier_signal
```
### Question1C : Create Bias Moving average in 5,10,20,60 days, RSI 14, DIF, MACD signal, MACD histogram & save dataframe to csv
```python
# In[1C] 
moving_avg_day = [5,10,20,60]
for day in moving_avg_day:
    TAIEX_df[f"MA{day}"] = bios_MA(TAIEX_df.Close, day)
# In[1C] 
TAIEX_df["RSI14"] = RSI(TAIEX_df.Close,14)
# In[1C]
EMA_12 = TAIEX_df['Close'].ewm(span=12, adjust=False).mean()
EMA_26 = TAIEX_df['Close'].ewm(span=26, adjust=False).mean()
TAIEX_df['DIF'] = EMA_12 - EMA_26
TAIEX_df['MACD_signal'] = TAIEX_df['DIF'].ewm(span=9, adjust=False).mean()
TAIEX_df['MACD_his'] = TAIEX_df['DIF']-TAIEX_df['MACD_signal']


TAIEX_df.to_csv("TAIEX_down.csv")
```

### Question2
```python
# In[2]
targets = np.array(TAIEX_df.TB_label[59:]) #extract label
TAIEX_array = np.array(TAIEX_df[["Open", "High", "Low", "Close", "Volume", 
                        "MA5", "MA10", "MA20", "MA60", "RSI14", 
                        "DIF", "MACD_signal", "MACD_his"]])

TAIEX_array = TAIEX_array[59:, :] #extract train

xtr = TAIEX_array[:int(0.7*len(targets)),:]
xte = TAIEX_array[int(0.7*len(targets)):,:]
ytr = targets[:int(0.7*len(targets))]
yte = targets[int(0.7*len(targets)):]
```



### Question2C : Print the distribution of all data, training data and testing data.
```python
# In[2C]
fig, ax = plt.subplots(1, 3, sharex = True, sharey=True, figsize = (12,8))
fig.subplots_adjust(hspace=0.2, wspace=0.2)
ax[0].hist(targets, density = True)
ax[0].set_title("all data")
ax[1].hist(ytr, density = True)
ax[1].set_title("training data")
ax[2].hist(yte, density = True)
ax[2].set_title("testing data")
plt.show()
```
![](https://i.imgur.com/WldEDEn.png)


### Question2D : Apply 3-fold cross validation and grid search to tune the hyperparameter
```python
# In[2D]
ytr = label_binarize(ytr, classes=[0, 1, 2])
yte = label_binarize(yte, classes=[0, 1, 2])
reg=RandomForestClassifier(n_jobs=-1,verbose=2)
param_grid={'bootstrap':[True],
            'max_depth':[80,90,100,110],
            'max_features': [2, 3],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 10, 12],
            'n_estimators':[100,200,300,1000]
            }
gs=GridSearchCV(reg,param_grid=param_grid,cv=3,verbose=-1) #grid search
gs.fit(xtr,ytr)
```

### Question2E : Print the best training score, best parameter and testing score of best estimator found from problem 2-D. The score is defined as accuracy
```python
# In[2E]
ytr_pre = gs.predict(xtr)
train_score = accuracy_score(ytr,ytr_pre)
print(f"best train accuracy = {train_score:2.2f}")
yte_pre = gs.predict(xte)
test_score = accuracy_score(yte,yte_pre)
print(f"best test accuracy = {test_score:2.2f}")
```
```python
# output
best train accuracy = 0.92
best test accuracy = 0.39
```

### Question2F : Plot the ROC curve of the best estimator found from problem 2-D
```python
# In[2D]
ytr = label_binarize(ytr, classes=[0, 1, 2])
yte = label_binarize(yte, classes=[0, 1, 2])
reg=RandomForestClassifier(n_jobs=-1,verbose=2)
param_grid={'bootstrap':[True],
            'max_depth':[80,90,100,110],
            'max_features': [2, 3],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 10, 12],
            'n_estimators':[100,200,300,1000]
            }
gs=GridSearchCV(reg,param_grid=param_grid,cv=3,verbose=-1) #grid search
gs.fit(xtr,ytr)
```
![](https://i.imgur.com/NrXrqbx.png)
