# run the simulation

# imports
import math 
import copy
import pandas as pd
import numpy as np
import helper.data as d
import model.lstm as model
from visualise import model_visualise as vis_mod
from visualise import visualise as vis


# import the data 
agile_price = d.get_data('data/Corr_Agile_Wholesale.xlsx')
# reformat the dates to remove errors in data
agile_price.iloc[:, 'date'] = agile_price['date'].apply(d.date_reformat)

# set data for this run 
### SET THE ZONE HERE ###
zone = ' LONDON'
data = agile_price[zone]
date = agile_price['date']

# some initial plots 
vis.plot_price(data, date, zone)
vis.hist(data, zone) # use default bins (20)

# plot price since 2023
vis.plot_price(data[63000:], date[63000:], zone)

# plot the log changes and histogram
vis.plot_log_changes(data, date)


##### ML PART #####

# set the model 
model = model.lstm_model(batch_size=32, window_size=20, features=1, verbose=True)

# create the datasets
data_np = data.values.astype("float32")
data_np = data_np.reshape(-1, 1)

# normalise the data
dataset = model.normalizer.fit_transform(data_np)

# set train data percentage 
train_percentage = 0.7

train_size = int(len(dataset) * train_percentage)
test_size = len(dataset) - train_size

train = dataset[0:train_size, :]
test = dataset[train_size: len(dataset), :]

print("Number of samples training set: " + str((len(train))))
print("Number of samples test set: " + str((len(test))))

window_size = 20

# create the dataset 
x_train, y_train = model.create_dataset(train, window_size=window_size)
x_test, y_test = model.create_dataset(test, window_size=window_size)

# reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

print("Shape of training inputs: " + str((x_train.shape)))
print("Shape of training labels: " + str((y_train.shape)))

# perform training 
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)

# get the score and predictions
train_score, train_pred = model.get_predict_and_score(x_train, y_train)
test_score, test_pred = model.get_predict_and_score(x_test, y_test)

print("Training data error: %.2f MSE" % train_score)
print("Test data error: %.2f MSE" % test_score)


# visulaise the results 
vis_mod.plot_all_train_test(train_pred, test_pred, dataset, date, model.normalizer, window_size)

# visualise a small section 
vis_mod.plot_all_train_test(train_pred[62900:63300], test_pred[62900:63300], dataset[62900:63300], date[62900:63300], model.normalizer, window_size)

# predict the next 24 hours
iterations = 48
predict_X = copy.deepcoy(x_test[-1:])

predict_X = np.reshape(predict_X, (1, predict_X.shape[1], 1))

predictions = model.predict(iterations, predict_X)

# plot predictions 
vis_mod.plot_predictions(predictions, zone) 

# predict 24 hours ahead and compare to real data
predict_X = copy.deepcopy(x_test[-48:-47])
predict_X = np.reshape(predict_X, (1, predict_X.shape[1], 1))

predictions = model.predict(iterations, predict_X)

# plot predictions 
vis_mod.plot_future_24(predictions, zone, date[-48:], data[-48:])


### DONE ###
