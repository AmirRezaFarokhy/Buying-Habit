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
			
		
tokens_value, tokens_key = TokenizingDateTimes()  
df['created_at'] = df['created_at'].map(tokens_value)

# we must split any user_id and product_id
data_dict = {}
for name, value in df.groupby(by=['user_id', 'product_id']):
	data_dict[name] = sorted(list(set(value['created_at'].values)))


model = LinearRegression(n_jobs=10)

# we predict next date with simple linear model for all value
for key in data_dict.keys():
	x_train = [i for i in range(1, len(data_dict[key])+1)]
	y_train = data_dict[key]
	x_train, y_train = np.array(x_train).reshape(-1, 1), np.array(y_train)
	model.fit(x_train, y_train)
	x_forcast = np.array(len(x_train)+1).reshape(-1, 1)
	y_future = model.predict(x_forcast)
	data_dict[key].append(y_future.tolist())


real_time = {}
for key, val in data_dict.items():
    real_time[key] = tokens_key[int(val[-1][0])]    # the last and prediction
