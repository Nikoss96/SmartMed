import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from data.paths import MEDIA_PATH, DATA_PATH, CLUSTER_ANALYSIS, USER_DATA_PATH, \
    ELBOW_METHOD, K_MEANS
from preprocessing.preprocessing import get_numeric_df


class ClusterModule:
    def __init__(self, df, chat_id):
        self.df = df
        self.chat_id = chat_id
        self.settings = {"fillna": "mean", "method": 0, "metric": 0,
                         "encoding": "label_encoding", "scaling": False}

    def elbow_method_and_optimal_clusters(self, max_clusters):
        inertia_values = []

        for n_clusters in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(self.df)
            inertia_values.append(kmeans.inertia_)

        plt.plot(range(1, max_clusters + 1), inertia_values, marker='o')
        plt.xlabel('Количество Кластеров')
        plt.ylabel('Инерция')
        plt.title('Метод Локтя')

        plt.savefig(
            f"{MEDIA_PATH}/{DATA_PATH}/{CLUSTER_ANALYSIS}/{ELBOW_METHOD}/elbow_method_{self.chat_id}.png",
        )

        plt.clf()
        plt.close()

        differences = [inertia_values[i] - inertia_values[i - 1] for i in
                       range(1, len(inertia_values))]
        norm_diff = differences / np.max(differences)

        elbow_cluster = np.argmax(
            norm_diff) + 1

        if elbow_cluster == 1:
            elbow_cluster = np.argmax(
                norm_diff[1:]) + 2

        return elbow_cluster

    def generate_k_means(self, num_clusters, file_path="hi.xlsx"):
        df_numeric = get_numeric_df(self.df)
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++')
        kmeans.fit(df_numeric)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        cluster_elements = [[] for _ in range(num_clusters)]
        for i, label in enumerate(labels):
            cluster_elements[label].append(i)

        df_cluster_assignments = pd.DataFrame(
            {'Кластеры': [i + 1 for i in range(num_clusters)],
             'Количество элементов': [len(elements) for elements in
                                      cluster_elements],
             'Элементы': [', '.join(map(str, elements)) for elements in
                          cluster_elements],
             },
        )

        df_cluster_assignments.to_excel(
            f"{MEDIA_PATH}/{DATA_PATH}/{CLUSTER_ANALYSIS}/{K_MEANS}/k_means_{self.chat_id}.xlsx",
            index=False)

        num_features = min(2, df_numeric.shape[1])
        feature_columns = np.random.choice(df_numeric.columns,
                                           size=num_features, replace=False)

        fig, ax = plt.subplots()
        colors = list(mcolors.TABLEAU_COLORS.keys())[:num_clusters]

        for i in range(num_clusters):
            if i < len(cluster_elements):
                cluster_data = df_numeric.iloc[cluster_elements[i]]
                ax.scatter(cluster_data[feature_columns[0]],
                           cluster_data[feature_columns[1]], c=colors[i],
                           label=f'Кластер №{i + 1}')

        ax.scatter(centers[:, df_numeric.columns.get_loc(feature_columns[0])],
                   centers[:, df_numeric.columns.get_loc(feature_columns[1])],
                   c='black', marker='x', label='Центроиды')

        ax.set_xlabel(f"Параметр {feature_columns[0]}")
        ax.set_ylabel(f"Параметр {feature_columns[1]}")
        ax.set_title('Визуализация кластеризации k-средними')
        ax.legend()
        plt.savefig(
            f"{MEDIA_PATH}/{DATA_PATH}/{CLUSTER_ANALYSIS}/{K_MEANS}/k_means_{self.chat_id}.png",
        )

        plt.clf()
        plt.close()
