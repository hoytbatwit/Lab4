import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from joblib import dump, load 

#chose linear regression as our model
model = LinearRegression(fit_intercept=True)

#read in data
pongData = pd.DataFrame(pd.read_csv("pong_data.csv"))

#arranging data
#features matrix
X_paddle = pongData.drop('paddle_direction', axis = 1)
#print(X_paddle.shape)
#target array
Y_paddle = np.array(pongData['paddle_direction'])
#making sure it has correct dimensions
Y_paddle = Y_paddle[:, np.newaxis]
#print(Y_paddle.shape)

#fitting data to our model
model.fit(X_paddle, Y_paddle)
dump(model, 'Pongmodel.joblib')
