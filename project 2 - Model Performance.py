import pandas as pd
import numpy as np
#from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt

#dirname = "C:/Documents/vt/ML 2022 SP/project2"
filename ="data_0.xlsx"
df = pd.read_excel(filename)
print('File {0} is size of {1}'.format(filename,df.shape))
labels = df.columns

# shift by specified # days and join to original dataframe
#deniz changed num_days to 0 here
num_days = 0
df1 = 0
dfout = df
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
#print(dfout)
print('New file is size of {0}'.format(dfout.shape))

labels = dfout.columns
targetlabel = ['Target']
unusedlabel = ['Date','Unnamed: 0']
featurelabels = labels.drop(targetlabel + unusedlabel)

features = dfout[featurelabels]
target = dfout[targetlabel]
X = features.values
y = target.values

# normalize features
scalerX = MinMaxScaler()
scalerX.fit(X)
X = scalerX.transform(X)

# normalize target
scaler_y = MinMaxScaler()
scaler_y.fit(y)
y_norm = scaler_y.transform(y)

# linear regression
rseed = 24061
(X_train,X_test,y_train,y_test) = train_test_split(X,y,test_size = 0.3, random_state = rseed)
# Train the model
lm = LinearRegression().fit(X_train,y_train)
# predict on the test data
y_pred = lm.predict (X_test)
MSE = mean_squared_error(y_test,y_pred)
r_sqr = r2_score(y_test,y_pred)
print('MSE of linear regression model is: {0}'.format(MSE))
print('R square value of linear regression model is: {0}'.format(r_sqr))

# neural network using MLP classifier
(X_m_train,X_m_test,y_m_train,y_m_test) = train_test_split(X,y_norm,test_size = 0.3, random_state = rseed)

hl = (3,3)
mlp = MLPClassifier(learning_rate='constant',tol = 0.0001,hidden_layer_sizes = hl,random_state=rseed,early_stopping=True)
mlp.fit(X_m_train,y_m_train)

