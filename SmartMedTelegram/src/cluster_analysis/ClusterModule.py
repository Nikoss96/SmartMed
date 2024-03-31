import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import plotly.graph_objects as go
from scipy.cluster._hierarchy import linkage
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, fcluster
import plotly.figure_factory as ff

from data.paths import (
    MEDIA_PATH,
    DATA_PATH,
    CLUSTER_ANALYSIS,
    USER_DATA_PATH,
    ELBOW_METHOD,
    K_MEANS,
    HIERARCHICAL,
)
from preprocessing.preprocessing import get_numeric_df


class ClusterModule:
    def __init__(self, df, chat_id):
        self.df = df
        self.chat_id = chat_id
        self.settings = {
            "fillna": "mean",
            "method": 0,
            "metric": 0,
            "encoding": "label_encoding",
            "scaling": False,
        }

    def elbow_method_and_optimal_clusters(self, max_clusters):
        inertia_values = []

        for n_clusters in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(self.df)
            inertia_values.append(kmeans.inertia_)

        plt.plot(range(1, max_clusters + 1), inertia_values, marker="o")
        plt.xlabel("Количество Кластеров")
        plt.ylabel("Инерция")
        plt.title("Метод локтя")

        plt.savefig(
            f"{MEDIA_PATH}/{DATA_PATH}/{CLUSTER_ANALYSIS}/{ELBOW_METHOD}/elbow_method_{self.chat_id}.png",
        )

        plt.clf()
        plt.close()

        differences = [
            inertia_values[i] - inertia_values[i - 1]
            for i in range(1, len(inertia_values))
        ]
        norm_diff = differences / np.max(differences)

        elbow_cluster = np.argmax(norm_diff) + 1

        if elbow_cluster == 1:
            elbow_cluster = np.argmax(norm_diff[1:]) + 2

        return elbow_cluster

    def generate_k_means(self, num_clusters):
        df_numeric = get_numeric_df(self.df)
        kmeans = KMeans(n_clusters=num_clusters, init="k-means++")
        kmeans.fit(df_numeric)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        cluster_elements = [[] for _ in range(num_clusters)]
        for i, label in enumerate(labels):
            cluster_elements[label].append(i)

        df_cluster_assignments = pd.DataFrame(
            {
                "Кластеры": [i + 1 for i in range(num_clusters)],
                "Количество элементов": [
                    len(elements) for elements in cluster_elements
                ],
                "Элементы": [
                    ", ".join(map(str, elements)) for elements in
                    cluster_elements
                ],
            },
        )

        df_cluster_assignments.to_excel(
            f"{MEDIA_PATH}/{DATA_PATH}/{CLUSTER_ANALYSIS}/{K_MEANS}/k_means_{self.chat_id}.xlsx",
            index=False,
        )

        num_features = min(2, df_numeric.shape[1])
        feature_columns = np.random.choice(
            df_numeric.columns, size=num_features, replace=False
        )

        fig, ax = plt.subplots()
        colors = list(mcolors.TABLEAU_COLORS.keys())[:num_clusters]

        for i in range(num_clusters):
            if i < len(cluster_elements):
                cluster_data = df_numeric.iloc[cluster_elements[i]]
                ax.scatter(
                    cluster_data[feature_columns[0]],
                    cluster_data[feature_columns[1]],
                    c=colors[i],
                    label=f"Кластер №{i + 1}",
                )

        ax.scatter(
            centers[:, df_numeric.columns.get_loc(feature_columns[0])],
            centers[:, df_numeric.columns.get_loc(feature_columns[1])],
            c="black",
            marker="x",
            label="Центроиды",
        )

        ax.set_xlabel(f"Параметр {feature_columns[0]}")
        ax.set_ylabel(f"Параметр {feature_columns[1]}")
        ax.set_title("Кластеризация методом к-средних")
        ax.legend()
        plt.savefig(
            f"{MEDIA_PATH}/{DATA_PATH}/{CLUSTER_ANALYSIS}/{K_MEANS}/k_means_{self.chat_id}.png",
        )

        plt.clf()
        plt.close()

    def plot_dendrogram(self, n_clusters=1):
        data = self.df.to_numpy()
        dists = np.zeros((data.shape[0] * (data.shape[0] - 1)) // 2)

        k = 0
        for i in range(data.shape[0]):
            for j in range(i + 1, data.shape[0]):
                dists[k] = np.linalg.norm(data[i] - data[j])
                k += 1

        linked = linkage(dists, n=data.shape[0], method=5)

        cluster_labels = fcluster(linked, n_clusters, criterion="maxclust")

        fig = ff.create_dendrogram(
            data,
            orientation="bottom",
            labels=self.df.index,
            linkagefun=lambda x: linked,
        )
        fig.update_layout(
            title=f"Дендрограмма",
            xaxis_title="Строка",
            yaxis_title="Евклидово расстояние",
        )

        for cluster_num in range(1, n_clusters + 1):
            cluster_points = self.df.index[cluster_labels == cluster_num]
            fig.add_trace(
                go.Scatter(
                    x=cluster_points,
                    y=np.zeros(len(cluster_points)),
                    mode="markers",
                    name=f"Кластер {cluster_num}",
                )
            )

        fig.update_layout(height=1000, width=1400)
        fig.update_xaxes(showticklabels=True)

        fig.write_image(
            f"{MEDIA_PATH}/{DATA_PATH}/{CLUSTER_ANALYSIS}/{HIERARCHICAL}/hierarchical_{self.chat_id}.png"
        )
