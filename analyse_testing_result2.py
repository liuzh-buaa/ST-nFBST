"""
    根据pdf表格中筛选出的传感器在图中高亮表示
"""
import os.path

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyse_testing_result import get_exp_id
from libcity.utils import ensure_dir

if __name__ == '__main__':
    plt.rc('font', family='Times New Roman', size=20)

    dataset = 'METR_LA'
    indices1 = [21, 26, 35, 50, 121, 126, 189, 200]
    indices2 = [0, 1, 9, 10, 54, 144, 184, 205]
    sensors = 207
    res_dir = 'analyse_testing_result2/METR_LA'

    ensure_dir(res_dir)

    filepath = f'libcity/cache/graph_sensor_locations.csv'
    df = pd.read_csv(filepath)

    mean_latitude = df['latitude'].mean()
    mean_longitude = df['longitude'].mean()

    m = folium.Map(location=(mean_latitude, mean_longitude), zoom_start=12)

    for data in df.iterrows():
        tmp_index = int(data[1]['index'])
        tmp_latitude = data[1]['latitude']
        tmp_longitude = data[1]['longitude']
        tmp_sensor_id = int(data[1]['sensor_id'])
        if tmp_index in indices1:
            icon = folium.Icon(color='green')
        elif tmp_index in indices2:
            icon = folium.Icon(color='red')
        else:
            icon = None
        folium.Marker(location=(tmp_latitude, tmp_longitude), tooltip=f'{tmp_index}',
                      popup=f'{tmp_sensor_id}:({tmp_latitude},{tmp_longitude})', icon=icon).add_to(m)

    m.save(f'{res_dir}/map_{dataset}.html')
