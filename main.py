import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def TokenizingDateTimes():
	date_to_token = {}
	token_to_date = {}
	tknz = 1
	for year in range(2019, 2021): # years create
		for month in range(1, 13): # month create
			for day in range(1, 32): # days create
				if day<10 and month<10:
					date_to_token[f"{year}-0{month}-0{day}"] = tknz
					token_to_date[tknz] = f"{year}-0{month}-0{day}"
					tknz += 1
				elif day<10 and month>=10:
					date_to_token[f"{year}-{month}-0{day}"] = tknz
					token_to_date[tknz] = f"{year}-{month}-0{day}"
					tknz += 1
				elif day>=10 and month<10:
					date_to_token[f"{year}-0{month}-{day}"] = tknz
					token_to_date[tknz] = f"{year}-0{month}-{day}"
					tknz += 1
				else:
					date_to_token[f"{year}-{month}-{day}"] = tknz
					token_to_date[tknz] = f"{year}-{month}-{day}"
					tknz += 1
					
	return date_to_token, token_to_date        

# values must be the same length
def MinimomLength(ln):
	return min([len(i) for i in ln])

# we must split any user_id and product_id
def SplitID()
	data_dict = {}
	for name, value in df.groupby(by=['user_id', 'product_id']):
	    data_dict[name] = sorted(list(set(value['created_at'].values)))
	return data_dict

df = pd.read_csv("purchase_history.csv")
tokens_value, tokens_key = TokenizingDateTimes()  
data_dict = SplitID()
chunke = MinimomLength(data_dict.values())

x_train = []
y_train = []
for val in data_dict.values():
    x_train.append(val[-chunke:-1])
    y_train.append(val[-1])
    
x_train = np.array(x_train)
y_train = np.array(y_train)
print(f"shape X is {x_train.shape}")
print(f"shape y is {y_train.shape}")

# the best model is DecisionTreeRegressor that show in jupyter notebook 97%
model = DecisionTreeRegressor()
model.fit(x_train, y_train)

print(f"accuracy for this model {model.score()}")
print(f"mean squared error for this model {mean_squared_error(y_train, model.predict(x_train))}")


# predict the example future
x_future = np.array([44, 106, 171]) 
forcast = model.predict(x_future.reshape(1, -1))
print(f"On this date {tokens_key[int(forcast)]}, a person buys.:)")


