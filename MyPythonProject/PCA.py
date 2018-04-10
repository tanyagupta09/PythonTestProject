# -*- coding: utf-8 -*-
"""
PCA on iris dataset
"""

import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

#load data iris into pandas data frame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
#a = pd.read_table('http://bit.ly/movieusers',sep='|',header=None)
print(df.head())

from sklearn.preprocessing import StandardScalar

features = ['sepal length','sepal width','petal length','petal width']
x = df.i[:,features].values
y = df.loc[:,['target']].values
x = StandardScalar().fit_transform(x)

print(x.head())