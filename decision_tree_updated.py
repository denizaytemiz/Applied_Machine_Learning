# -*- coding: utf-8 -*-
"""decision_tree_updated.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SN55KyRUYaMc5iHMjqqiMdtwGrKmyhJ1
"""

# for calculating MSE and R square values
import pandas 
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

### Import data and normalize
filename = "data_0.xlsx"
df = pandas.read_excel(filename, index_col=0)  # read an Excel spreadsheet
print('File {0} is of size {1}'.format(filename, df.shape))

df_date = df['Date']
df.drop(['Date'],axis=1,inplace=True)


transformer = MinMaxScaler().fit(df)
transformerT = MinMaxScaler().fit(df['Close'].to_numpy().reshape(-1, 1))

out = transformer.transform(df)

dfout = pandas.DataFrame(out, columns = df.columns)
dfout = dfout.join(df_date)

### Shift data
# shift by specified # days and join to original dataframe
num_days = 10
df1 = 0
df = dfout
for n in range(1,num_days+1):
    df1 = df.shift(periods = n).add_suffix(str(n))
    dfout = dfout.join(df1)
    df1 = 0

# drop rows that now have NA values
dfout.drop(dfout.index[range(num_days)], axis=0, inplace=True)

# create target column
t = 28 # number days ahead to predict
dfout['Target'] = dfout['Close'].shift(periods = -t)
dfout.reset_index(inplace=True)
dfout.drop(['index'], axis=1, inplace=True)

# again drop rows that now have NA values
dfout.drop(dfout.index[range(-1,(-t-1),-1)], axis=0, inplace=True)
df_date = dfout['Date']

### prep for learning
df = dfout
labels = df.columns
targetlabel = ['Target']
unusedlabel = ['Date']
for n in range(num_days):
    temp = "Date" + str(n+1)
    unusedlabel.append(temp)

featurelabels = labels.drop(targetlabel + unusedlabel)

features = df[featurelabels]
target = df[targetlabel]
X = features.values
y = target.values

rseed = 24061
error_df1 = pandas.DataFrame(columns = ['max_depth','MSE','Rsquare'])
error_df2 = pandas.DataFrame(columns = ['max_depth','MSE','Rsquare'])
#best split trees are modeled and MSE and R2 values are saved.
for i in range(10):
### Start learning
# Train the model
  row=[]
  (X_train,X_test,y_train,y_test) = train_test_split(X,y,test_size = 0.3, random_state = rseed)
  tree = DecisionTreeRegressor(criterion='squared_error', max_depth=i+1,random_state=None, splitter='best').fit(X_train, y_train)


# predict on the test data
  y_pred = tree.predict(X_test)
  row=[format(i+1)]
  row.append(mean_squared_error(y_test,y_pred))
  row.append(r2_score(y_test,y_pred))
  error_df1.loc[i] = row
  error_df1.to_excel("best split.xlsx")
### End learning

for i in range(10):
### Start learning
# Train the model
  row=[]
  (X_train,X_test,y_train,y_test) = train_test_split(X,y,test_size = 0.3, random_state = rseed)
  tree = DecisionTreeRegressor(criterion='squared_error', max_depth=i+1, random_state=0).fit(X_train, y_train)


# predict on the test data
  y_pred = tree.predict(X_test)
  row=[format(i+1)]
  row.append(mean_squared_error(y_test,y_pred))
  row.append(r2_score(y_test,y_pred))
  error_df2.loc[i] = row
  error_df2.to_excel("random split.xlsx")
### End learning


### End learning

# finding investment gain for best splitter&random splitter models
import pandas 
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

### Import data and normalize
filename = "data_0.xlsx"
df = pandas.read_excel(filename, index_col=0)  # read an Excel spreadsheet
print('File {0} is of size {1}'.format(filename, df.shape))

df_date = df['Date']
df.drop(['Date'],axis=1,inplace=True)


transformer = MinMaxScaler().fit(df)
transformerT = MinMaxScaler().fit(df['Close'].to_numpy().reshape(-1, 1))

out = transformer.transform(df)

dfout = pandas.DataFrame(out, columns = df.columns)
dfout = dfout.join(df_date)

### Shift data
# shift by specified # days and join to original dataframe
num_days = 10
df1 = 0
df = dfout
for n in range(1,num_days+1):
    df1 = df.shift(periods = n).add_suffix(str(n))
    dfout = dfout.join(df1)
    df1 = 0

# drop rows that now have NA values
dfout.drop(dfout.index[range(num_days)], axis=0, inplace=True)

# create target column
t = 28 # number days ahead to predict
dfout['Target'] = dfout['Close'].shift(periods = -t)
dfout.reset_index(inplace=True)
dfout.drop(['index'], axis=1, inplace=True)

# again drop rows that now have NA values
dfout.drop(dfout.index[range(-1,(-t-1),-1)], axis=0, inplace=True)
df_date = dfout['Date']

### prep for learning
df = dfout
labels = df.columns
targetlabel = ['Target']
unusedlabel = ['Date']
for n in range(num_days):
    temp = "Date" + str(n+1)
    unusedlabel.append(temp)

featurelabels = labels.drop(targetlabel + unusedlabel)

features = df[featurelabels]
target = df[targetlabel]
X = features.values
y = target.values

rseed = 24061


for i in range(10):
### Start learning
# Train the model
  row=[]
  (X_train,X_test,y_train,y_test) = train_test_split(X,y,test_size = 0.3, random_state = rseed)
  tree = DecisionTreeRegressor(criterion='squared_error', max_depth=i+1, random_state=0).fit(X_train, y_train)
  # run on full dataset
  out_pred = tree.predict(X)
  df_pred = pandas.DataFrame(data=out_pred)
  df_tar = pandas.DataFrame(data=target)
  df_close = pandas.DataFrame(data=df['Close'])

  out = transformerT.inverse_transform(df_pred)
  out2 = transformerT.inverse_transform(df_tar)
  out3 = transformerT.inverse_transform(df_close)
  dfout = pandas.DataFrame(out, columns=['predicted future closing price'])
  dfout2 = pandas.DataFrame(out2, columns=['real future closing price'])
  dfout3 = pandas.DataFrame(out3, columns=['current closing price'])
  df_final = dfout.join(dfout2)
  df_final = df_final.join(dfout3)
  df_final = df_final.join(df_date)
  ### End learning
  ## Estimate total gains or losses
  investment = 1000
  df_final['diff'] = df_final['predicted future closing price'] - df_final['real future closing price']
  df_final.loc[df_final['diff'] <= 0, 'invest'] = 0
  df_final.loc[df_final['diff'] > 0, 'invest'] = 1
  df_final['amount'] = ((investment//df_final['current closing price'])*df_final['real future closing price']-investment)*df_final['invest'] # can only purchase full shares
  print('Investing $',investment, 'every time the closing price', t, 'days later is predicted'
                                                                     ' to be greater than the current closing price results in a total gain (or loss if negative) of about: $',round(df_final['amount'].sum()))