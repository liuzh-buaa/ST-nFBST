import numpy as np
import pandas as pd

from analyse_testing_result6 import get_metr_la_adjacency_matrix
from visualize_sensor import visualize_sensor

if __name__ == '__main__':

    for dataset in ['METR_LA', 'PEMS_BAY']:
        if dataset == 'METR_LA':
            distance = 4000
            threshold = 6
        else:
            distance = 4000
            threshold = 5
        adj_mx = get_metr_la_adjacency_matrix(dataset)
        count_below_distance0 = np.sum(adj_mx <= distance, axis=0)
        count_below_distance1 = np.sum(adj_mx <= distance, axis=1)
        central0 = np.where(count_below_distance0 >= threshold)[0]
        marginal0 = np.where(count_below_distance0 < threshold)[0]
        central1 = np.where(count_below_distance1 >= threshold)[0]
        marginal1 = np.where(count_below_distance1 < threshold)[0]
        central = list(set(central0) | set(central1))
        marginal = list(set(marginal0) & set(marginal1))

        visualize_sensor(dataset, marginal, central)
