# main.py

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate,cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from util import fill_x_prediction_data, fill_data_by_month, fill_zones, fill_x_y_data, fill_zone_station_map, fill_prediction_data, \
     fill_prediction_error_data,  merge_data

# load data
load_data = pd.read_csv('../Load_history_final.csv')
temperature_data = pd.read_csv('../Temp_history_final.csv')

# add extra columns 'date' and 'h_avg' in order to analyzing correlation easily
temperature_data["date"] = temperature_data["year"].map(str)+"-" + temperature_data["month"].map(str)+"-" + temperature_data["day"].map(str)
load_data["date"] = load_data["year"].map(str)+"-" + load_data["month"].map(str)+"-" + load_data["day"].map(str)

temperature_data['h_avg'] = temperature_data.loc[:, 'h1':'h24'].mean(axis=1)
load_data['h_avg'] = load_data.loc[:, 'h1':'h24'].mean(axis=1)
# ========================================data pre-processing============================================ #
# get prediction dataset
df_prediction = fill_x_prediction_data(temperature_data)
# only use subset of five months around June, because correlation depends on the season.
# for winter, the correlation is negative (low temperature -> high load),
# for summer, the correlation is positive (high temperature -> high load)
max_year = 2008
summer_months = [5, 6, 7, 8, 9]
load_data_summer = fill_data_by_month(data=load_data, max_year=max_year, months=summer_months)
temperature_data_summer = fill_data_by_month(data=temperature_data, max_year=max_year, months=summer_months)

# ========================================correlation analysis============================================ #
# select mean of 24 hours data to combine zone and station data,
merged_data = merge_data(df_zone=load_data_summer, df_station=temperature_data_summer, hour='', is_average=True)
print(merged_data.head(5))
# calculate correlations between zones and stations with spearman method
df_corr = merged_data.corr(method='pearson')
df_corr = abs(df_corr)
zone_columns = fill_zones(load_data_summer)
df_corr = df_corr[zone_columns]
df_corr = df_corr.drop(zone_columns)
print(df_corr)
# get max corr for each zone
df_zones = pd.concat([df_corr.idxmax(), df_corr.max()], axis=1).reset_index().rename(
    columns={'index': 'zone', 0: 'station', 1: 'corr'})

df_zones.to_csv('./map_of_zone_and_station_with_max_corr.csv')
# remove low corr map, only keep the map which corr > 0.6
df_zones = df_zones[df_zones['corr'] > 0.6]
print(df_zones)
# get zone-station map
zone_station_map = fill_zone_station_map(df_zones=df_zones)

# --------------------------------Use MLPRegressor model#
table_output_data = list()
table_prediction_error = list()
table_prediction_result = list()
SCORE_COLUMNS=['zone','station','fit_train_time','train_score','test_score']
ERROR_COLUMNS=['zone_id','year','month','day','hour','y_pred_load','y_true_load','relative percentage error (%)']
LOAD_COLUMNS = ['zone_id','year','month','day','h1','h2','h3','h4','h5','h6','h7','h8','h9','h10','h11','h12','h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23','h24']

#Question 2
for zs_map in zone_station_map:
    print('zone, station: ', zs_map)
    output = list()
    output.append(zs_map[0])
    output.append(zs_map[1])
    X, y, df_X, df_y = fill_x_y_data(df_zone=load_data_summer, df_station=temperature_data_summer,
                                     zone_station_map=[zs_map])
    
    #Question 4
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20, random_state=3)
    df_X_train_val, df_X_test, df_y_train_val, df_y_test = train_test_split(df_X, df_y, test_size=0.20, random_state=3)

    #train model
    tic = time.perf_counter()
    best_model = MLPRegressor(hidden_layer_sizes=(10,),
                              learning_rate_init=0.1,
                              max_iter=100)
    
    #Question 5 and #Question 7
    cv_results = cross_validate(best_model, X_train_val, y_train_val, cv=3, return_train_score=True)
    sorted(cv_results.keys())
    train_score = np.mean(cv_results['train_score'])
    #val_score = np.mean(cv_results['test_score'])
    test_score = np.mean(cv_results['test_score'])
    test_predict = cross_val_predict(best_model, X_train_val, y_train_val, cv=3)

    for index, row_values in enumerate(df_y_test.values):
        for i in range(4, 28):
            prediction_error = fill_prediction_error_data(row_values, test_predict, index, i)
            table_prediction_error.append(prediction_error)
    toc = time.perf_counter()

    output.append(round(toc - tic, 4))
    output.append(round(train_score, 2))
    #output.append(round(val_score, 2))
    output.append(round(test_score, 2))
    table_output_data.append(output)

best_model = 'MLPRegressor'
df_output = pd.DataFrame(data=table_output_data, columns=SCORE_COLUMNS)
print(df_output)
df_output.to_csv('./train_time_and_score_' + str(best_model) + '.csv')

df_prediction_error = pd.DataFrame(data=table_prediction_error, columns=ERROR_COLUMNS)
df_prediction_error = df_prediction_error.sort_values(by='relative percentage error (%)', ascending=False,ignore_index=True)
df_prediction_error = df_prediction_error.head(10)
print(df_prediction_error)
df_prediction_error.to_csv('./top_10_prediction_errors_' + str(best_model) + '.csv')

# --------------------------Use LinearRegression model#
table_output_data = list()
table_prediction_error = list()
table_prediction_result = list()
SCORE_COLUMNS=['zone','station','fit_train_time','train_score','test_score']
ERROR_COLUMNS=['zone_id','year','month','day','hour','y_pred_load','y_true_load','relative percentage error (%)']
LOAD_COLUMNS = ['zone_id','year','month','day','h1','h2','h3','h4','h5','h6','h7','h8','h9','h10','h11','h12','h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23','h24']

#Question 2
for zs_map in zone_station_map:
    print('zone, station: ', zs_map)
    output = list()
    output.append(zs_map[0])
    output.append(zs_map[1])
    X, y, df_X, df_y = fill_x_y_data(df_zone=load_data_summer, df_station=temperature_data_summer,
                                     zone_station_map=[zs_map])

    #Question 4
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20, random_state=3)
    df_X_train_val, df_X_test, df_y_train_val, df_y_test = train_test_split(df_X, df_y, test_size=0.20, random_state=3)

    #train model
    tic = time.perf_counter()
    best_model = LinearRegression(n_jobs=10)
    
    #Question 5 and #Question 7
    cv_results = cross_validate(best_model, X_train_val, y_train_val, cv=3, return_train_score=True)
    sorted(cv_results.keys())
    train_score = np.mean(cv_results['train_score'])
    #val_score = np.mean(cv_results['test_score'])
    test_score = np.mean(cv_results['test_score'])
    test_predict = cross_val_predict(best_model, X_train_val, y_train_val, cv=3)
    for index, row_values in enumerate(df_y_test.values):
        for i in range(4, 28):
            prediction_error = fill_prediction_error_data(row_values, test_predict, index, i)
            table_prediction_error.append(prediction_error)
    toc = time.perf_counter()
    output.append(round(toc - tic, 4))
    output.append(round(train_score, 2))
    #output.append(round(val_score, 2))
    output.append(round(test_score, 2))
    table_output_data.append(output)

    # predict hourly load values for June 1-7, 2008
    X_prediction, df_prediction_zone = fill_prediction_data(df_prediction=df_prediction, zone_station_map=[zs_map])
    tic = time.perf_counter()
    best_model.fit(X_train_val, y_train_val)
    prediction_results = best_model.predict(X_prediction)
    toc = time.perf_counter()
    print(f"The time of prediction is  {toc - tic:0.4f} seconds")
    for index, row_values in enumerate(df_prediction_zone.values):
        prediction_result = list()
        prediction_result.append(row_values[0])
        prediction_result.append(row_values[1])
        prediction_result.append(row_values[2])
        prediction_result.append(row_values[3])
        for result in prediction_results[index]:
            prediction_result.append(round(result, 0))
        table_prediction_result.append(prediction_result)

best_model = 'LinearRegression'
df_output = pd.DataFrame(data=table_output_data, columns=SCORE_COLUMNS)
print(df_output)
df_output.to_csv('./train_time_and_score_' + str(best_model) + '.csv')

df_prediction_error = pd.DataFrame(data=table_prediction_error, columns=ERROR_COLUMNS)
df_prediction_error = df_prediction_error.sort_values(by='relative percentage error (%)', ascending=False, ignore_index=True)
df_prediction_error = df_prediction_error.head(10)
print(df_prediction_error)
df_prediction_error.to_csv('./top_10_prediction_errors_' + str(best_model) + '.csv')

df_load_prediction = pd.DataFrame(data=table_prediction_result, columns=LOAD_COLUMNS)
print(df_load_prediction)
df_load_prediction.to_csv('./load_prediction.csv')
