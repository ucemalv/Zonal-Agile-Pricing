"""mean error simulation"""

import helper.data as d
import model.lstm as model 
import copy
from tqdm import tqdm
import numpy as np
import math
from sklearn.metrics import mean_squared_error 


# import data 
agile_price = d.import_data('data/Corr_Agile_Wholesale.xlsx')
# reformat data
agile_price.loc[:, 'date'] = agile_price['date'].apply(d.date_reformat)

#### PICK ZONE ID ####
zone_id = 6
zone_name = agile_price.keys()[zone_id]

data = agile_price[zone_name]

model = model.load_model("model_name_here")

data_np = data.values.astype("float32")
data_np = data_np.reshape(-1,1)

# normalise data
dataset = model.normalizer.fit_transform(data_np)

window_size = 20
X, Y = model.create_dataset(data, window_size=window_size)

iterations = 48
tests = 500
prediction_scores = []

for i in tqdm(range(tests)):
  random_number = np.random.randint(0, len(X)-48)
  # extract this data point 
  predict_X = copy.deepcopy(X[-random_number:-random_number+1])
  # reshape 
  predict_X = np.reshape(predict_X, (1, predict_X.shape[1], 1))

  # do the predictions loop
  predictions = model.predict(iterations, predict_X)

  # calc MSE
  score = math.sqrt(mean_squared_error(data[-random_number:-random_number+48], predictions))
  prediction_scores.append(score)

print("Successfully ran all predictions")

average = sum(prediction_scores) / len(prediction_scores)
print(f"Over {iterations} iterations, {zone_name} has a Mean Squared Error of: {average}.")
