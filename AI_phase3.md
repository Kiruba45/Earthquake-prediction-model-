earthquake-prediction

October 17, 2023

> **1** **Earthquake Prediction**
>
> It is well known that if a disaster has happened in a region, it is
> likely to happen there again. Some regions really have frequent
> earthquakes, but this is just a comparative quantity compared to other
> regions. So, predicting the earthquake with Date and Time, Latitude
> and Longitude from previous data is not a trend which follows like
> other things, it is natural occuring.
>
> Import the necessary libraries required for buidling the model and
> data analysis of the earthquakes.

  -----------------------------------------------------------------------
  \[ \]: **import numpy as np**\
  **import pandas as pd**\
  **import matplotlib.pyplot as plt**
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

+-----------------------------------------------------------------------+
| > **import os**\                                                      |
| > print(os.listdir(\"../input\"))                                     |
+=======================================================================+
+-----------------------------------------------------------------------+

> Read the data from csv and also columns which are necessary for the
> model and the column which needs to be predicted.

+-----------------------------------------------------------------------+
|   ------------------------------------------------------------------  |
|   \[ \]: data = pd.read_csv(\"../input/database.csv\") data.head()    |
|   ------------------------------------------------------------------  |
|                                                                       |
|   ------------------------------------------------------------------  |
+=======================================================================+
+-----------------------------------------------------------------------+

\[ \]: data.columns

> Figure out the main features from earthquake data and create a object
> of that features, namely, Date, Time, Latitude, Longitude, Depth,
> Magnitude.

  -----------------------------------------------------------------------
  \[ \]: data = data\[\[\'Date\', \'Time\', \'Latitude\', \'Longitude\',
  \'Depth\', \'Magnitude\'\]\] data.head()
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

> Here, the data is random we need to scale according to inputs to the
> model. In this, we convert given Date and Time to Unix time which is
> in seconds and a numeral. This can be easily used as input for the
> network we built.

  -----------------------------------------------------------------------
  \[ \]: **import datetime**\
  **import time**
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

+-----------------------------------------------------------------------+
| > timestamp = \[\]\                                                   |
| > **for** d, t **in** zip(data\[\'Date\'\], data\[\'Time\'\]):        |
+=======================================================================+
+-----------------------------------------------------------------------+

1

+-----------------------------------------------------------------------+
| +------------------------------------------------------------------+  |
| | > **try**:\                                                      |  |
| | > ts = datetime.datetime.strptime(d+\' \'+t, \'%m/**%d**/%Y      |  |
| | > %H:%M:%S\') timestamp.append(time.mktime(ts.timetuple()))\     |  |
| | > **except ValueError**:\                                        |  |
| | > *\# print(\'ValueError\')*\                                    |  |
| | > timestamp.append(\'ValueError\')                               |  |
| +==================================================================+  |
| +------------------------------------------------------------------+  |
+=======================================================================+
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
|   ------------------------------------------------------------------  |
|   \[ \]: timeStamp = pd.Series(timestamp) data\[\'Timestamp\'\] =     |
|   timeStamp.values                                                    |
|   ------------------------------------------------------------------  |
|                                                                       |
|   ------------------------------------------------------------------  |
+=======================================================================+
+-----------------------------------------------------------------------+

+-----------------------------------+-----------------------------------+
|   -                               |                                   |
| --------------------------------- |                                   |
|   \[ \]: final_data =             |                                   |
|                                   |                                   |
| data.drop(\[\'Date\', \'Time\'\], |                                   |
|   axis=1)\                        |                                   |
|   final_data =                    |                                   |
|                                   |                                   |
|  final_data\[final_data.Timestamp |                                   |
|   != \'ValueError\'\]             |                                   |
|   final_data.head()               |                                   |
|   -                               |                                   |
| --------------------------------- |                                   |
|                                   |                                   |
|   -                               |                                   |
| --------------------------------- |                                   |
+===================================+===================================+
| **1.1**                           | > **Visualization**               |
+-----------------------------------+-----------------------------------+

> Here, all the earthquakes from the database in visualized on to the
> world map which shows clear representation of the locations where
> frequency of the earthquake will be more.

  -----------------------------------------------------------------------
  \[ \]: **from mpl_toolkits.basemap import** Basemap
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

+-----------------------------------------------------------------------+
| > m = Basemap(projection=\'mill\',llcrnrlat=-80,urcrnrlat=80,␣\       |
| > ↪llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution=\'c\')\          |
| > longitudes = data\[\"Longitude\"\].tolist()\                        |
| > latitudes = data\[\"Latitude\"\].tolist()\                          |
| > *#m = Basemap(width=12000000,height=9000000,projection=\'lcc\',*\   |
| > *#resolution=None,lat_1=80.,lat_2=55,lat_0=80,lon_0=-107.)* x,y =   |
| > m(longitudes,latitudes)                                             |
+=======================================================================+
+-----------------------------------------------------------------------+

+-----------------------------------+-----------------------------------+
|   ----------------------          |                                   |
| --------------------------------- |                                   |
|   \[ \]: f                        |                                   |
| ig = plt.figure(figsize=(12,10))\ |                                   |
|   p                               |                                   |
| lt.title(\"All affected areas\")\ |                                   |
|   m.plot(x, y, \"o\",             |                                   |
| markersize = 2, color = \'blue\') |                                   |
|   m.drawcoastlines()\             |                                   |
|   m.fillcontinents(col            |                                   |
| or=\'coral\',lake_color=\'aqua\') |                                   |
|   m.drawmapboundary()\            |                                   |
|   m.drawcountries()\              |                                   |
|   plt.show()                      |                                   |
|   ----------------------          |                                   |
| --------------------------------- |                                   |
|                                   |                                   |
|   ----------------------          |                                   |
| --------------------------------- |                                   |
+===================================+===================================+
| **1.1.1**                         | > **Splitting the Data**          |
+-----------------------------------+-----------------------------------+

> Firstly, split the data into Xs and ys which are input to the model
> and output of the model respectively. Here, inputs are TImestamp,
> Latitude and Longitude and outputs are Magnitude and Depth. Split the
> Xs and ys into train and test with validation. Training dataset
> contains 80% and Test dataset contains 20%.

2

  -----------------------------------------------------------------------
  \[ \]: X = final_data\[\[\'Timestamp\', \'Latitude\', \'Longitude\'\]\]
  y = final_data\[\[\'Magnitude\', \'Depth\'\]\]
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

  -----------------------------------------------------------------------
  \[ \]: **from sklearn.cross_validation import** train_test_split\
  X_train, X_test, y_train, y_test = train_test_split(X, y,
  test_size=0.2,␣print(X_train.shape, X_test.shape, y_train.shape,
  X_test.shape)↪random_state=42)
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

> Here, we used the RandomForestRegressor model to predict the outputs,
> we see the strange pre-diction from this with score above 80% which
> can be assumed to be best fit but not due to its predicted values.

  -----------------------------------------------------------------------
  \[ \]: **from sklearn.ensemble import** RandomForestRegressor
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

+-----------------------------------------------------------------------+
| > reg = RandomForestRegressor(random_state=42) reg.fit(X_train,       |
| > y_train)\                                                           |
| > reg.predict(X_test)                                                 |
+=======================================================================+
+-----------------------------------------------------------------------+

\[ \]: reg.score(X_test, y_test)

  -----------------------------------------------------------------------
  \[ \]: **from sklearn.model_selection import** GridSearchCV
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

+-----------------------------------------------------------------------+
| > parameters = {\'n_estimators\':\[10, 20, 50, 100, 200, 500\]}       |
+=======================================================================+
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| > grid_obj = GridSearchCV(reg, parameters) grid_fit =                 |
| > grid_obj.fit(X_train, y_train) best_fit =                           |
| > grid_fit.best_estimator\_\                                          |
| > best_fit.predict(X_test)                                            |
+=======================================================================+
+-----------------------------------------------------------------------+

\[ \]: best_fit.score(X_test, y_test)

+-----------------------------------+-----------------------------------+
| **1.1.2**                         | > **Neural Network model**        |
+===================================+===================================+
+-----------------------------------+-----------------------------------+

> In the above case it was more kind of linear regressor where the
> predicted values are not as expected. So, Now, we build the neural
> network to fit the data for training set. Neural Network consists of
> three Dense layer with each 16, 16, 2 nodes and relu, relu and softmax
> as activation function.

  -----------------------------------------------------------------------
  \[ \]: **from keras.models import** Sequential\
  **from keras.layers import** Dense
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

+-----------------------------------------------------------------------+
| > **def** create_model(neurons, activation, optimizer, loss):\        |
| > model = Sequential()\                                               |
| > model.add(Dense(neurons, activation=activation, input_shape=(3,)))  |
| > model.add(Dense(neurons, activation=activation))\                   |
| > model.add(Dense(2, activation=\'softmax\'))                         |
+=======================================================================+
+-----------------------------------------------------------------------+

3

+-----------------------------------------------------------------------+
| > model.compile(optimizer=optimizer, loss=loss,                       |
| > metrics=\[\'accuracy\'\])                                           |
+=======================================================================+
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| > **return** model                                                    |
+=======================================================================+
+-----------------------------------------------------------------------+

> In this, we define the hyperparameters with two or more options to
> find the best fit.

  -----------------------------------------------------------------------
  \[ \]: **from keras.wrappers.scikit_learn import** KerasClassifier
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

+-----------------------------------------------------------------------+
| > model = KerasClassifier(build_fn=create_model, verbose=0)           |
| >                                                                     |
| > *\# neurons = \[16, 64, 128, 256\]*\                                |
| > neurons = \[16\]\                                                   |
| > *\# batch_size = \[10, 20, 50, 100\]*\                              |
| > batch_size = \[10\]\                                                |
| > epochs = \[10\]\                                                    |
| > *\# activation = \[\'relu\', \'tanh\', \'sigmoid\',                 |
| > \'hard_sigmoid\', \'linear\',*␣                                     |
| >                                                                     |
| > activation = \[\'sigmoid\', \'relu\'\]↪*\'exponential\'\]*\         |
| > *\# optimizer = \[\'SGD\', \'RMSprop\', \'Adagrad\', \'Adadelta\',  |
| > \'Adam\', \'Adamax\',*␣                                             |
| >                                                                     |
| > optimizer = \[\'SGD\', \'Adadelta\'\]↪*\'Nadam\'\]*\                |
| > loss = \[\'squared_hinge\'\]                                        |
+=======================================================================+
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| > param_grid = dict(neurons=neurons, batch_size=batch_size,           |
| > epochs=epochs,␣                                                     |
+=======================================================================+
+-----------------------------------------------------------------------+

> ↪activation=activation, optimizer=optimizer, loss=loss)\
> Here, we find the best fit of the above model and get the mean test
> score and standard deviation of the best fit model.

  -----------------------------------------------------------------------
  \[ \]: grid = GridSearchCV(estimator=model, param_grid=param_grid,
  n_jobs=-1) grid_result = grid.fit(X_train, y_train)
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

+-----------------------------------------------------------------------+
| > print(\"Best: **%f** using **%s**\" % (grid_result.best_score\_,    |
| > grid_result.best_params\_)) means =                                 |
| > grid_result.cv_results\_\[\'mean_test_score\'\]\                    |
| > stds = grid_result.cv_results\_\[\'std_test_score\'\]\              |
| > params = grid_result.cv_results\_\[\'params\'\]\                    |
| > **for** mean, stdev, param **in** zip(means, stds, params):\        |
| > print(\"**%f** (**%f**) with: **%r**\" % (mean, stdev, param))      |
+=======================================================================+
+-----------------------------------------------------------------------+

> The best fit parameters are used for same model to compute the score
> with training data and testing data.

  -----------------------------------------------------------------------
  \[ \]: model = Sequential()\
  model.add(Dense(16, activation=\'relu\', input_shape=(3,)))
  model.add(Dense(16, activation=\'relu\'))\
  model.add(Dense(2, activation=\'softmax\'))
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

+-----------------------------------------------------------------------+
| > model.compile(optimizer=\'SGD\', loss=\'squared_hinge\',            |
| > metrics=\[\'accuracy\'\])                                           |
+=======================================================================+
+-----------------------------------------------------------------------+

4

  -----------------------------------------------------------------------
  \[ \]: model.fit(X_train, y_train, batch_size=10, epochs=20,
  verbose=1,␣ ↪validation_data=(X_test, y_test))
  -----------------------------------------------------------------------
  \[ \]: \[test_loss, test_acc\] = model.evaluate(X_test, y_test)\
  print(\"Evaluation result on Test Data : Loss = **{}**, accuracy =
  **{}**\".

  -----------------------------------------------------------------------

> ↪format(test_loss, test_acc))
>
> We see that the above model performs better but it also has lot of
> noise (loss) which can be
>
> neglected for prediction and use it for furthur prediction.
>
> The above model is saved for furthur prediction.

\[ \]: model.save(\'earthquake.h5\')

5
