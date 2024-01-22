# https://blog.csdn.net/qq_40206371/article/details/134698358
import folium
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex


def visualize_sensor(dataset, key_index=None, indices=None, filename=None, ext=None):
    if dataset.lower() == 'metr_la':
        filepath = f'libcity/cache/graph_sensor_locations.csv'
        df = pd.read_csv(filepath)
    else:
        filepath = 'libcity/cache/graph_sensor_locations_bay.csv'
        df = pd.read_csv(filepath, names=['sensor_id', 'latitude', 'longitude'])
        df.insert(0, 'index', range(len(df)), allow_duplicates=False)

    if indices is None:
        indices = range(len(df))

    if key_index is None:
        key_index = []

    mean_latitude = df['latitude'].mean()
    mean_longitude = df['longitude'].mean()

    m = folium.Map(location=(mean_latitude, mean_longitude), zoom_start=12)

    for data in df.iterrows():
        tmp_index = int(data[1]['index'])
        if tmp_index not in indices and tmp_index not in key_index:
            continue
        tmp_latitude = data[1]['latitude']
        tmp_longitude = data[1]['longitude']
        tmp_sensor_id = int(data[1]['sensor_id'])
        icon = folium.Icon(color='red') if tmp_index in key_index else None
        testing_result = '' if ext is None else ext[tmp_index]
        folium.Marker(location=(tmp_latitude, tmp_longitude), tooltip=f'{tmp_index}',
                      popup=f'{tmp_sensor_id}={testing_result}:({tmp_latitude},{tmp_longitude})', icon=icon).add_to(m)

    if filename is None:
        m.save(f'map_{dataset}.html')
    else:
        m.save(filename)


def visualize_sensor_marked(dataset, key_index, filename=None, ext=None, scaled=0.1):
    """只突出标记最不显著的scaled比例的传感器"""
    if dataset.lower() == 'metr_la':
        filepath = f'libcity/cache/graph_sensor_locations.csv'
        df = pd.read_csv(filepath)
    else:
        filepath = 'libcity/cache/graph_sensor_locations_bay.csv'
        df = pd.read_csv(filepath, names=['sensor_id', 'latitude', 'longitude'])
        df.insert(0, 'index', len(df), allow_duplicates=False)

    mean_latitude = df['latitude'].mean()
    mean_longitude = df['longitude'].mean()

    sorted_items = sorted(ext.items(), key=lambda x: x[1])
    indices = [item[0] for item in sorted_items[-int(scaled * len(sorted_items)):]]

    m = folium.Map(location=(mean_latitude, mean_longitude), zoom_start=12)

    for data in df.iterrows():
        tmp_index = int(data[1]['index'])
        tmp_latitude = data[1]['latitude']
        tmp_longitude = data[1]['longitude']
        tmp_sensor_id = int(data[1]['sensor_id'])
        if tmp_index == key_index:
            folium.Marker(location=(tmp_latitude, tmp_longitude), tooltip=f'{tmp_index}',
                          popup=f'{tmp_sensor_id}={ext[tmp_index]}:({tmp_latitude},{tmp_longitude})',
                          icon=folium.Icon(color='red')).add_to(m)
            # folium.CircleMarker(location=[tmp_latitude, tmp_longitude],
            #                     radius=10,
            #                     color='blue',
            #                     fill=True,
            #                     fill_color='blue',
            #                     fill_opacity=1,
            #                     tooltip=f'{tmp_index}',
            #                     popup=f'{tmp_sensor_id}={ext[tmp_index]}:({tmp_latitude},{tmp_longitude})').add_to(m)
        elif tmp_index in indices:
            folium.CircleMarker(location=[tmp_latitude, tmp_longitude],
                                radius=10,
                                color='red',
                                fill=True,
                                fill_color='red',
                                fill_opacity=1,
                                tooltip=f'{tmp_index}',
                                popup=f'{tmp_sensor_id}={ext[tmp_index]}:({tmp_latitude},{tmp_longitude})').add_to(m)
        else:
            folium.CircleMarker(location=[tmp_latitude, tmp_longitude],
                                radius=10,
                                color='blue',
                                fill=True,
                                fill_color='blue',
                                fill_opacity=1,
                                tooltip=f'{tmp_index}',
                                popup=f'{tmp_sensor_id}={ext[tmp_index]}:({tmp_latitude},{tmp_longitude})').add_to(m)

    if filename is None:
        m.save(f'map_{dataset}.html')
    else:
        m.save(filename)


def visualize_sensor_varying(dataset, key_index, ext, filename=None, adjust=False, scaled=None, speeds=None, normalized=False):
    if dataset.lower() == 'metr_la':
        filepath = f'libcity/cache/graph_sensor_locations.csv'
        df = pd.read_csv(filepath)
    else:
        filepath = 'libcity/cache/graph_sensor_locations_bay.csv'
        df = pd.read_csv(filepath, names=['sensor_id', 'latitude', 'longitude'])
        df.insert(0, 'index', len(df), allow_duplicates=False)

    mean_latitude = df['latitude'].mean()
    mean_longitude = df['longitude'].mean()

    # if adjust:
    #     sorted_items = sorted(ext.items(), key=lambda x: x[1])
    #     minVal = 1.1 * sorted_items[1][1] - 0.1 * sorted_items[-1][1]
    #     assert minVal > 0, f'diff={sorted_items[-1][1]}-{sorted_items[1][1]}'
    #     ext[sorted_items[0][0]] = minVal
    #     diff = max(ext.values()) - minVal
    #     ext = {key: 1 - (value - minVal) / diff for key, value in ext.items()}

    if adjust:
        sorted_items = sorted(ext.items(), key=lambda x: x[1])
        minVal = sorted_items[0][1]
        maxVal = 1.1 * sorted_items[-2][1] - 0.1 * minVal
        ext[sorted_items[-1][0]] = maxVal
        diff = maxVal - minVal
        ext = {key: value - minVal / diff for key, value in ext.items()}

    if scaled is not None:
        sorted_items = sorted(ext.items(), key=lambda x: x[1])
        n = len(sorted_items)
        scaled_n = int(scaled * n)
        for i in range(scaled_n):
            ext[sorted_items[i][0]] = 0
        for i in range(scaled_n):
            ext[sorted_items[n - i - 1][0]] = 1
        minVal = sorted_items[scaled_n][1]
        maxVal = sorted_items[n - scaled_n - 1][1]
        for i in range(scaled_n, n - scaled_n):
            ext[sorted_items[i][0]] = (sorted_items[i][1] - minVal) / (maxVal - minVal)

    if normalized:
        minVal = min(ext.values())
        maxVal = max(ext.values())
        for k, v in ext.items():
            ext[k] = (v - minVal) / (maxVal - minVal) + minVal

    # 获取Viridis颜色映射的颜色列表
    viridis_colors = [to_hex(c) for c in plt.cm.viridis(range(256))]
    plasma_colors = [to_hex(c) for c in plt.cm.plasma(range(256))]
    inferno_colors = [to_hex(c) for c in plt.cm.inferno(range(256))]
    magma_colors = [to_hex(c) for c in plt.cm.magma(range(256))]
    # 创建颜色映射，使用Viridis颜色映射
    colormap = folium.LinearColormap(colors=viridis_colors, vmin=min(ext.values()), vmax=max(ext.values()))
    # colormap = folium.LinearColormap(colors=['blue', 'red'], vmin=min(ext.values()), vmax=max(ext.values()))
    m = folium.Map(location=(mean_latitude, mean_longitude), zoom_start=13)

    for data in df.iterrows():
        tmp_index = int(data[1]['index'])
        tmp_latitude = data[1]['latitude']
        tmp_longitude = data[1]['longitude']
        tmp_sensor_id = int(data[1]['sensor_id'])
        if tmp_index not in ext.keys():
            print(f'Jump Sensor {tmp_index}.')
            continue
        if tmp_index == key_index:
            folium.CircleMarker(location=[tmp_latitude, tmp_longitude],
                                radius=20,
                                color='red',
                                fill=True,
                                fill_color='red',
                                fill_opacity=1,
                                tooltip=f'{tmp_index}',
                                popup=f'{tmp_sensor_id}={ext[tmp_index]}:({tmp_latitude},{tmp_longitude})').add_to(m)

            # folium.Marker(location=(tmp_latitude, tmp_longitude), tooltip=f'{tmp_index}',
            #               popup=f'{tmp_sensor_id}={ext[tmp_index]}:({tmp_latitude},{tmp_longitude})',
            #               radius=20,
            #               icon=folium.Icon(color='red')).add_to(m)

            # folium.CircleMarker(location=[tmp_latitude, tmp_longitude],
            #                     radius=10,
            #                     color='blue',
            #                     fill=True,
            #                     fill_color='blue',
            #                     fill_opacity=1,
            #                     tooltip=f'{tmp_index}',
            #                     popup=f'{tmp_sensor_id}={ext[tmp_index]}:({tmp_latitude},{tmp_longitude})').add_to(m)
        else:
            popup_content = '' if speeds is None else f':{speeds[tmp_index]}'
            folium.CircleMarker(location=[tmp_latitude, tmp_longitude],
                                radius=20,
                                color=colormap(ext[tmp_index]),
                                fill=True,
                                fill_color=colormap(ext[tmp_index]),
                                fill_opacity=1,
                                tooltip=f'{tmp_index}',
                                popup=f'{tmp_sensor_id}={ext[tmp_index]}:({tmp_latitude},{tmp_longitude}){popup_content}').add_to(m)

    # Add LinearColormap to the map
    colormap.caption = 'Bayesian Evidence'
    m.add_child(colormap)

    # Add LayerControl to show/hide the color bar
    folium.LayerControl().add_to(m)

    if filename is None:
        m.save(f'map_{dataset}.html')
    else:
        m.save(filename)


if __name__ == '__main__':
    visualize_sensor('metr_la')
    visualize_sensor('pems_bay')
