# util.py

import numpy as np
import pandas as pd
from pandas import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor


def join_date(year, month, day):
    return '-'.join([str(year), str(month).zfill(2), str(day).zfill(2)])


def fill_x_prediction_data(temperature_data):
    x_prediction = temperature_data[temperature_data['year'] == 2008]
    x_prediction = x_prediction[x_prediction['month'] == 6]
    x_prediction = x_prediction[x_prediction['day'] <= 7]
    return x_prediction


def fill_data_by_month(data: DataFrame, max_year: int, months: list):
    result_data = data[data['year'] < max_year]
    if len(months) > 0:
        result_data = result_data[result_data['month'].isin(months)]
    return result_data



def fill_zones(df_zone: DataFrame):
    zones = df_zone['zone_id'].unique()
    zone_columns = []
    for zone in zones:
        column_name = 'zone_' + str(zone)
        zone_columns.append(column_name)
    return zone_columns


def fill_zone_station_map(df_zones: DataFrame):
    zone_station_map = list()
    for item in df_zones.values:
        zone = item[0].replace('zone_', '')
        zone = int(zone)
        station = item[1].replace('station_', '')
        station = int(station)
        zone_station_map.append([zone, station])
    return zone_station_map


def fill_x_y_data(df_zone: DataFrame, df_station: DataFrame, zone_station_map: list):
    df_x = pd.DataFrame(columns=df_station.columns)
    df_y = pd.DataFrame(columns=df_zone.columns)
    for zs_map in zone_station_map:
        load_data_by_zone = df_zone[df_zone['zone_id'] == zs_map[0]]
        temperature_data_by_station = df_station[df_station['station_id'] == zs_map[1]]
        df_y = pd.concat([df_y, load_data_by_zone])
        df_x = pd.concat([df_x, temperature_data_by_station])

    x = df_x.loc[:, 'h1':'h24']
    x = x.values
    x = x.astype(np.float)
    y = df_y.loc[:, 'h1':'h24']
    y = y.values
    y = y.astype(np.float)
    return x, y, df_x, df_y


def fill_prediction_data(df_prediction: DataFrame, zone_station_map: list):
    df_p = pd.DataFrame(columns=df_prediction.columns)
    for zs_map in zone_station_map:
        load_data_by_station = df_prediction[df_prediction['station_id'] == zs_map[1]]
        df_p = pd.concat([df_p, load_data_by_station])

    p = df_p.loc[:, 'h1':'h24']
    p = p.values
    p = p.astype(np.float)
    return p, df_p



def fill_prediction_error_data(row_values, test_predict, row_index, column_index):
    prediction_error = list()
    prediction_error.append(row_values[0])
    prediction_error.append(row_values[1])
    prediction_error.append(row_values[2])
    prediction_error.append(row_values[3])
    prediction_error.append('h' + str(column_index - 3))
    #pred_load
    y_pred = test_predict[row_index][column_index - 4]
    prediction_error.append(round(y_pred, 0))
    #true_load
    y_true = row_values[column_index]
    prediction_error.append(y_true)
    error = round(abs((y_true - y_pred) / y_true) * 100, 2)
    prediction_error.append(error)
    return prediction_error





def merge_data(df_zone: DataFrame, df_station: DataFrame, hour: str, is_average=False):
    zones = df_zone['zone_id'].unique()
    stations = df_station['station_id'].unique()
    merged_data = pd.DataFrame(data=df_zone['date'].unique(), columns=['date'], dtype='str')
    for zone in zones:
        column_name = 'zone_' + str(zone)
        if is_average:
            merged_data[column_name] = df_zone[df_zone['zone_id'] == zone]['h_avg'].values
        else:
            merged_data[column_name] = df_zone[df_zone['zone_id'] == zone][hour].values
    for station in stations:
        column_name = 'station_' + str(station)
        if is_average:
            merged_data[column_name] = df_station[df_station['station_id'] == station]['h_avg'].values
        else:
            merged_data[column_name] = df_station[df_station['station_id'] == station][hour].values
    return merged_data