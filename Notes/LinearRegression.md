![](image/vdvv.jpg))

# 线性回归(LinearRegression)

在统计学中，简单(一元)线性回归是一个具有单一解释变量的线性回归模型。也就是说，它涉及一个自变量和一个因变量的二维采样点（通常是笛卡尔坐标系中的x和y坐标），并找到一个尽可能精确的线性函数（非垂直直线）。预测因变量值作为自变量的函数。

![](image/1569728547(1).png)

导入依赖库
```python
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
```

读取数据并展示
```python
data = pd.read_csv('Simple Linear Regression.csv')
data.head()
```
![](fvrevr.png)

staff_wage_mean 是我们的x变量，shanghai_GDP是目标变量。

将数据分割成训练集和测试集
```python
train, test = train_test_split(data, test_size=0.25, random_state=27)
```

模型在训练集上训练并在测试集测试
```python
lr = linear_model.LinearRegression()
lr.fit(train.staff_wage_mean.values.reshape(-1,1),train.shanghai_GDP.values.reshape(-1,1))
pred = lr.predict(test.staff_wage_mean.values.reshape(-1,1))
```
